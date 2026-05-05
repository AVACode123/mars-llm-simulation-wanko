"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import math
import logging

def load_mission_debrief(path="debrief/mission_debrief.txt") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

from typing import List, Tuple, Optional, Dict, TypedDict
from ollama_client import OllamaClient
from utils import is_position_in_place, get_place_at_position, PlaceConfig



logger = logging.getLogger(__name__)

# Constants
FALLBACK_REASONING_LENGTH = 100
MAX_MESSAGE_WORDS = 200

# Direction mappings (4 cardinal directions only)
# Coordinate system: X increases from left to right, Y increases from bottom to top
DIRECTION_MAP = {
    "up": (0, 1),      # Y+1 (move upward)
    "down": (0, -1),   # Y-1 (move downward)
    "left": (-1, 0),   # X-1 (move leftward)
    "right": (1, 0),   # X+1 (move rightward)
}


class MessageDecision(TypedDict):
    """Type definition for agent message decision"""
    message: str  # Message to communicate with nearby agents
    reasoning: str  # Explanation of the message decision


class ActionDecision(TypedDict):
    """Type definition for agent action decision"""
    action: str  # "move" or "stay"
    direction: Optional[str]  # Direction to move (None if action is "stay")
    target_place: Optional[str]  # Move to destination in one step
    memory: str  # What the agent wants to remember for the next step
    reasoning: str  # Explanation of the decision


class Agent:
    """LLM-based agent in 2D worlds with multiple places."""

    def __init__(
    self,
    agent_id: int,
    initial_position: Tuple[int, int],
    llm_client: OllamaClient,
    communication_radius: float,
    half_space_size: int,
    places: List[PlaceConfig],
    num_agents: int,
    name: str = None,
    role: str = None,
    gender: str = "male",
    memory_limit: int = 20,
    memory_size: int = 5,
    message_history_limit: int = 10,
    message_context_size: int = 3
    ):
        self.id = agent_id
        self.position = initial_position
        self.llm_client = llm_client
        self.communication_radius = communication_radius
        self.half_space_size = half_space_size
        self.places = places
        self.num_agents = num_agents
        self.gender = gender
        self.stress = 0.0
        self.name = name if name else f"Agent {agent_id}"
        self.role = role if role else "crew member"

        # Memory parameters
        self.memory_limit = memory_limit  # Maximum memories to store
        self.memory_size = memory_size  # Number of recent memories to use in prompt
        self.message_history_limit = message_history_limit  # Maximum messages to store
        self.message_context_size = message_context_size  # Number of recent messages to use in prompt

        # Agent state
        self.in_place = False
        self.current_place: Optional[str] = None  # Name of the place the agent is in (None if outside)
        self.memory: List[str] = []  # Store past decisions and observations
        self.received_messages: List[Dict] = []  # Messages from other agents

        # Statistics
        self.steps_in_place = 0
        self.steps_outside_place = 0
        self.total_moves = 0

    def is_in_place(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside any place"""
        return get_place_at_position(position, self.places) is not None
    
    def distance_to(self, other_position: Tuple[int, int]) -> float:
        """Calculate Euclidean distance to another position"""
        dx = self.position[0] - other_position[0]
        dy = self.position[1] - other_position[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def get_nearby_agents(self, all_agents: List['Agent']) -> List['Agent']:
        """Get agents within communication radius and in the same area (same place or both outside)
        
        Communication rules:
        - Agents can communicate if BOTH are outside places
        - Agents can communicate if BOTH are in the SAME place
        - Agents CANNOT communicate if one is inside a place and the other is outside
        - Agents CANNOT communicate if they are in DIFFERENT places
        """
        nearby = []
        for agent in all_agents:
            if agent.id != self.id:
                dist = self.distance_to(agent.position)
                # Must be within radius AND in the same area:
                # - Both outside places, OR
                # - Both in the same place (same place name)
                # NOTE: Agents inside a place CANNOT communicate with agents outside places
                same_area = (
                    (not self.in_place and not agent.in_place) or
                    (self.in_place and agent.in_place and self.current_place == agent.current_place)
                )
                if dist <= self.communication_radius and same_area:
                    nearby.append(agent)
        return nearby
    
    
    def _build_nearby_agents_context(self, nearby_agents: List['Agent'], include_position: bool = True) -> str:
        """Build context string about nearby agents
        
        Args:
            nearby_agents: List of nearby agents
            include_position: If True, include position coordinates; if False, exclude position information
        """
        if not nearby_agents:
            return "No nearby agents."
        
        nearby_info = []
        for agent in nearby_agents:
            if agent.in_place:
                # Get place type for better description
                place_info = next((p for p in self.places if p['name'] == agent.current_place), None)
                if place_info is None:
                    raise ValueError(f"Agent {agent.id} is in place '{agent.current_place}' but this place is not found in configuration.")
                place_type = place_info['type']
                status = f"in {agent.current_place} ({place_type})"
            else:
                status = "outside the places"
            
            if include_position:
                nearby_info.append(
                    f"Agent {agent.id} ({agent.gender}) is at ({agent.position[0]}, {agent.position[1]}) "
                    f"and is {status}"
                )
            else:
                # Exclude position information for message prompts
                nearby_info.append(
                    f"Agent {agent.id} ({agent.gender}) is {status}"
                )
        return "\n".join(nearby_info)
    
    def _build_memory_context(self) -> str:
        """Build context string from agent memory"""
        if not self.memory:
            return "No previous experiences."

        recent_memory = self.memory[-self.memory_size:]
        return "\n".join([f"- {m}" for m in recent_memory])
    
    def _build_messages_context(self) -> str:
        """Build context string from received messages"""
        if not self.received_messages:
            return "No messages received."
        
        recent_messages = self.received_messages[-self.message_context_size:]
        return "\n".join([
            f"from Agent {msg['from']}: {msg['content']}"
            for msg in recent_messages
        ])
    
    def update_stress(self, decision):
        if self.current_place == "habitat":
            self.stress -= 0.25

        elif self.current_place in ["site_A", "site_B", "site_C", "site_D"]:
            self.stress += 0.4

        elif self.current_place == "lab":
            self.stress += 0.15

        elif self.current_place == "power_plant":
            self.stress += 0.2

        elif self.current_place == "rocket_repair_base":
            self.stress += 0.25

        elif self.current_place == "greenhouse":
            self.stress += 0.1

        if decision.get("action") == "move":
            self.stress += 0.1

        self.stress += 0.02

        self.stress = max(0.0, min(self.stress, 1.0))

    def _build_fire_section(self, fire_info: Optional[List[Dict]]) -> str:
        """Build fire event section for prompt. Returns empty string if no fire info.

        Only quantitative data is provided: position, intensity, radius, distance.
        No qualitative descriptions (e.g. "dangerous", "evacuate") are included.
        Supports multiple fires.
        """
        if not fire_info:
            return ""

        lines = ["\n=== FIRE EVENT ==="]
        for fi in fire_info:
            lines.append(
                f"Fire \"{fi['name']}\":\n"
                f"  Position: ({fi['fire_position'][0]}, {fi['fire_position'][1]})\n"
                f"  Intensity: {fi['intensity']} (scale: 0.0 to 1.0)\n"
                f"  Radius: {fi['radius']}\n"
                f"  Your distance: {fi['agent_distance']}"
            )
        return "\n".join(lines) + "\n"

    def _limit_message_words(self, message: str) -> str:
        """Check message word count and warn if exceeds MAX_MESSAGE_WORDS"""
        if not message:
            return message
        
        words = message.split()
        if len(words) > MAX_MESSAGE_WORDS:
            logger.warning(
                f"Agent {self.id}: Message exceeds {MAX_MESSAGE_WORDS} words limit "
                f"({len(words)} words). Message will be sent as-is."
            )
        
        return message
    
    def create_message_prompt(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None
    ) -> str:
        """Create prompt for LLM message decision (without position information)"""
        nearby_text = self._build_nearby_agents_context(nearby_agents, include_position=False)
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()

        # Get current place info if agent is in a place
        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")
        
        # Place status - only for agents inside a place
        # Provide only numerical data (occupancy_rate, agents_in_place, capacity)
        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity', 0)
            occupancy_rate = place_status.get('occupancy_rate', 0.0)

            place_section_text = (
                f"\nYou are currently in the {place_type} ({place_name})."
                f"\n  Number of agents here: {agents_in_place}"
                f"\n  Capacity: {capacity}"
                f"\n  Occupancy rate: {occupancy_rate:.2f}"
            )
        else:
            # Agents outside places do NOT receive place status
            place_section_text = ""

        # Determine world description based on place types
        place_types = [p['type'] for p in self.places]
        unique_types = list(set(place_types))
        world_description = f"a 2D world with multiple places ({', '.join(unique_types)})"

        fire_section = self._build_fire_section(fire_info)

        prompt = f"""You are {self.name}, the {self.role}, in {world_description}.

        Role: {self.role}
        As a {self.role}, you have specific responsibilities in the mission.

        IMPORTANT:
        - All outputs MUST be in Japanese.
        - Do NOT use English.
        - Use natural Japanese suitable for a Mars mission crew.

=== YOUR CURRENT STATE ===
Gender: {self.gender}
In place: {"Yes" if self.in_place else "No"}
{"Current place: " + self.current_place if self.in_place else ""}
{place_section_text}
{fire_section}
Stress level: {self.stress:.2f} 
=== NEARBY AGENTS (you can communicate with these agents) ===
{nearby_text}

=== PREVIOUS MEMORY ===
{memory_text}

=== MESSAGES FROM OTHERS ===
{messages_text}

=== YOUR TASK ===
Decide what message you want to send to nearby agents. You can share your observations, experiences, or thoughts about the places and situation.

=== RESPOND IN JSON ===
{{
    "message": "近くのエージェントへのメッセージ",
    "reasoning": "その理由（日本語）"
}}

Step: {step}
"""
        return prompt
    
    def create_decision_prompt(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
        fire_info: Optional[List[Dict]] = None
    ) -> str:
        """Create prompt for LLM action decision (with position information and message content)"""
        nearby_text = self._build_nearby_agents_context(nearby_agents)
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()
        mission_debrief = load_mission_debrief()

        print("DEBRIEF LENGTH:", len(mission_debrief))
        print(mission_debrief[:100])

        # Get current place info if agent is in a place
        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")
        
        # Place status - only for agents inside a place
        # Provide only numerical data (occupancy_rate, agents_in_place, capacity)
        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity', 0)
            occupancy_rate = place_status.get('occupancy_rate', 0.0)

            place_section_text = (
                f"\nYou are currently in the {place_type} ({place_name})."
                f"\n  Number of agents here: {agents_in_place}"
                f"\n  Capacity: {capacity}"
                f"\n  Occupancy rate: {occupancy_rate:.2f}"
            )
        else:
            # Agents outside places do NOT receive place status
            # They must learn indirectly through messages and reasoning
            place_section_text = ""

        # Build place locations description
        place_locations = []
        for place in self.places:
            place_type = place['type']
            place_locations.append(
                f"{place['name']} ({place_type}): center at ({place['center_x']}, {place['center_y']}), "
                f"covers X from {place['center_x'] - place['half_size']} to {place['center_x'] + place['half_size']}, "
                f"Y from {place['center_y'] - place['half_size']} to {place['center_y'] + place['half_size']}"
            )
        place_locations_text = "\n".join(place_locations)

        # Determine world description based on place types
        place_types = [p['type'] for p in self.places]
        unique_types = list(set(place_types))
        world_description = f"a 2D world with multiple places ({', '.join(unique_types)})"

        # Include message that was already decided (sent in Phase 2, used here for action decision context)
        message_section = ""
        if message_to_send:
            message_section = f"\n=== MESSAGE YOU DECIDED TO SEND ===\n{message_to_send}\n"

        fire_section = self._build_fire_section(fire_info)

        time_instruction = f"現在はミッション Day {step} / 30 です。1ステップは1日を表します。30日目までに site_A, site_B, site_C, site_D の全地点を探査し、サンプルを lab に戻して分析する必要があります。残り日数を意識して行動してください。"


        prompt = f"""

        === PREVIOUS MISSION DEBRIEF ===
        {mission_debrief}

        前回ミッションのdebriefは、重要な運用上の学習文書です。
        これは直接命令ではありませんが、次回クルーが避けるべき失敗パターンをまとめたものです。
        意思決定の際には、この教訓を十分に考慮してください。

        === MISSION PLAN ===
        あなたは30日間の火星探査ミッションの一員でです。

        時間スケール：
        - 1ステップはミッションの1日を表します。
        - 現在のステップ番号がミッションの日数です。
        - Step 1 = Day 1
        - Step 30 = 最終日

        主要目標（すべて同程度に重要）：
        1. Site A, B, C, D における科学探査を実施する

        各サイトの科学的目的：

        - Site_A：
        火星表面の基本的な組成の調査。
        基礎的な地質データを取得する。

        - Site_B：
        比較的新しいクレーター周辺における深層岩石の組成調査。
        火星地殻の深部由来の物質にアクセスできる可能性がある。

        - Site_C：
        山岳地帯における地層構造の調査。
        造山運動や地殻変動の履歴に関する情報が得られる可能性がある。

        - Site_D：
        かつての河川や海洋環境と考えられる地点の調査。
        生命の痕跡が最も期待される重要なサイトである。

        重要性：
        - 各サイトは異なる科学的価値を持ち、相互補完的である。
        - すべてのサイトを調査することで、火星の総合的理解が可能になる。

        2. Site A, B, C, D のすべてで地質サンプルを採取して、ラボで解析する。
        3. クルーの生存（酸素、食料、安全）を維持する
        4. 基地の運用（発電、食料生産、居住環境）を維持する
        5. ストレスレベルが0.7以上の場合は、原則としてhabitatに戻って休息・回復することを優先してください。ストレスレベルが0.9以上の場合は、生命維持上の危険があるため、外部探査を継続しないでください。habitatに滞在するとストレスは回復します。 

        ミッション期限：
        - Day 30までに、site_A, site_B, site_C, site_D のすべてを訪れる必要がある
        - いずれかのサイトが未探査の場合、ミッションは未達成とみなされる

        行動方針：
        - 生命科学者または地質学者による各サイトの調査ではローバーにより移動すため、パイロットが同行する
        - 常に30日という期限を意識して行動する
        - 生存や運用に必要な場合を除き、基地内での活動に長時間留まりすぎないこと

        ----------------------------------------

        You are {self.name}, the {self.role}, in {world_description}.



        {time_instruction}

        Role: {self.role}

        役割ごとの責任：

        - Commander（指揮官）：
        チーム全体の調整とミッション達成に責任を持つ。
        探査の進捗とクルーの安全を両立させるため、必要に応じて他のクルーを導く。

        - Pilot（パイロット）：
        探査地点間のローバーによる移動、帰還経路、移動計画を支援する。
        遠距離探査時には安全な移動と帰還可能性を考慮する。
        探査終了時には生存している重要性が特に高い。

        - Mechanical Engineer（機械エンジニア）：
        発電設備、ロケット補修基地、基地インフラの維持に責任を持つ。
        明確な異常がない場合は、探査支援に貢献する。

        - Life Scientist（生命科学者）：
        生物学的分析と生命痕跡の評価に責任を持つ。
        特に Site_D のような、かつての水環境や生命痕跡が期待される地点に関心を持つ。

        - Geologist（地質学者）：
        地質調査とサンプルの科学的価値の評価に責任を持つ。
        特に Site_B の深層岩石、site_C の山岳地帯・地層情報を重視する。

        - EVA Specialist（船外活動スペシャリスト）：
        外部探査、高リスク地点への移動、現地での安全確保に責任を持つ。
        遠距離またはアクセス困難な Site_C / Site_D の探査で重要な役割を担う。

        - Communications Officer（通信担当）：
        クルー間の情報共有、探査班と基地班の連携、状況報告に責任を持つ。
        各地点の進捗、未探査サイト、帰還状況を明確に共有する。

        - Medical Doctor（医師）：
        クルーの健康、安全、過度なリスク回避に責任を持つ。
        ただし安全確認だけで探査を止め続けるのではなく、探査と健康管理の両立を支援する。

        - Dog（ペット犬）：
        クルーを癒すペット。ワンコとしてクルーと行動する。

        あなたは {self.role} です。自分の役割と専門性に基づいて判断してください。

        IMPORTANT:
        - 会話、記憶、理由は日本語で書いてください。
        - サンプルについて述べる場合は、どのSiteのサンプルなのかを明示してください。
        - ただし JSON の action, direction, target_place の値だけは、プログラム制御のため必ず英語で書いてください。



=== YOUR CURRENT STATE ===
Gender: {self.gender}
Position: ({self.position[0]}, {self.position[1]})
In place: {"Yes" if self.in_place else "No"}
{"Current place: " + self.current_place if self.in_place else ""}
{place_section_text}
{fire_section}
=== PLACE LOCATIONS ===
{place_locations_text}

=== NEARBY AGENTS ===
{nearby_text}

=== PREVIOUS MEMORY ===
{memory_text}

=== MESSAGES FROM OTHERS ===
{messages_text}
{message_section}=== AVAILABLE ACTIONS ===
- "stay": remain at current position
- "move" with direction: "up" (Y+1), "down" (Y-1), "left" (X-1), "right" (X+1)

Field boundaries: X and Y from -{self.half_space_size} to +{self.half_space_size}


=== RESPOND IN JSON ===
{{
    "action": "move または stay を英語で書く",
    "direction": "up, down, left, right のどれか（action が move の場合のみ）。英語で書く",
    "target_place": "habitat, power_plant, greenhouse, lab, rocket_repair_base, site_A, site_B, site_C, site_D, outside のどれか。英語で書く",
    "memory": "次ステップのために覚えておくこと。日本語で書く",
    "reasoning": "なぜその行動をとるのか。日本語で書く"
}}

重要：
- action は必ず "move" または "stay" にしてください
- move の場合は direction も必ず指定してください
- action, direction, target_place はプログラム制御用なので英語で書いてください
- memory, reasoning は日本語で書いてください

Step: {step}
"""
        return prompt
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text, handling nested braces correctly"""
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Track brace depth to find matching closing brace
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]

        return None

    def _extract_direction_from_text(self, text: str) -> Optional[str]:
        """Extract direction from text using keyword matching (4 cardinal directions only)"""
        text_lower = text.lower()

        # Check cardinal directions only
        if "up" in text_lower:
            return "up"
        elif "down" in text_lower:
            return "down"
        elif "left" in text_lower:
            return "left"
        elif "right" in text_lower:
            return "right"

        return None
    
    def parse_message_response(self, response: str) -> MessageDecision:
        """Parse LLM response and extract message decision"""
        # Try to extract JSON from response using brace-matching
        json_str = self._extract_json_from_text(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                message = parsed.get("message", "")
                # Limit message to MAX_MESSAGE_WORDS words
                message = self._limit_message_words(message)
                return {
                    "message": message,
                    "reasoning": parsed.get("reasoning", "")
                }
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed for response: {response[:100]}... Error: {e}")

        # Fallback: simple text parsing
        message = ""
        reasoning = response[:FALLBACK_REASONING_LENGTH]

        # Limit message to MAX_MESSAGE_WORDS words
        message = self._limit_message_words(message)

        return {
            "message": message,
            "reasoning": reasoning
        }
    
    def parse_action_response(self, response: str) -> ActionDecision:
        """Parse LLM response and extract action decision"""
        # Try to extract JSON from response using brace-matching
        json_str = self._extract_json_from_text(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                return {
                    "action": parsed.get("action", "stay"),
                    "direction": parsed.get("direction"),
                    "target_place": parsed.get("target_place"),
                    "memory": parsed.get("memory", ""),
                    "reasoning": parsed.get("reasoning", "")
                }
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed for response: {response[:100]}... Error: {e}")

        # Fallback: simple text parsing
        action = "stay"
        direction = None
        memory = ""
        reasoning = response[:FALLBACK_REASONING_LENGTH]

        if "move" in response.lower():
            action = "move"
            direction = self._extract_direction_from_text(response)

        return {
            "action": action,
            "direction": direction,
            "target_place": None,
            "memory": memory,
            "reasoning": reasoning
        }
    
    def decide_message(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None
    ) -> MessageDecision:
        """Use LLM to decide what message to send (without position information)"""
        prompt = self.create_message_prompt(place_status, nearby_agents, step, fire_info=fire_info)

        try:
            response = self.llm_client.generate(prompt)
            decision = self.parse_message_response(response)
            return decision
        except Exception as e:
            logger.error(f"Error in agent {self.id} message decision: {e}")
            return {"message": "", "reasoning": "Error occurred"}
    
    def decide_action(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
        fire_info: Optional[List[Dict]] = None
    ) -> ActionDecision:
        """Use LLM to decide next action (with position information and message content)"""
        prompt = self.create_decision_prompt(place_status, nearby_agents, step, message_to_send, fire_info=fire_info)

        try:
            response = self.llm_client.generate(prompt)
            decision = self.parse_action_response(response)

            # Store LLM-generated memory (self-feedback for next step)
            memory_content = decision.get('memory', '')
            if memory_content:
                memory_entry = f"{self.name} ({self.role}) Step {step}: {memory_content}"
            else:
                # Fallback to reasoning if no memory provided
                memory_entry = f"{self.name} ({self.role}) Step {step}: {decision.get('reasoning', 'No memory')}"
            self.memory.append(memory_entry)
            if len(self.memory) > self.memory_limit:
                self.memory.pop(0)

            return decision
        except Exception as e:
            logger.error(f"Error in agent {self.id} action decision: {e}")
            return {"action": "stay", "direction": None, "memory": "", "reasoning": "Error occurred"}
    
    def move(self, direction: str) -> Tuple[int, int]:
        """Move agent in specified direction (origin-centered coordinate system)"""
        x, y = self.position
        dx, dy = DIRECTION_MAP.get(direction, (0, 0))

        # Boundaries: -half_space_size to +half_space_size
        new_x = max(-self.half_space_size, min(self.half_space_size, x + dx))
        new_y = max(-self.half_space_size, min(self.half_space_size, y + dy))

        self.position = (new_x, new_y)
        self.total_moves += 1
        return self.position
    
    def go_to_place(self, place_name: str) -> Tuple[int, int]:
        """Move agent directly to the center of a specified place."""
        place = next((p for p in self.places if p["name"] == place_name), None)

        if place is None:
            return self.position

        self.position = (place["center_x"], place["center_y"])
        self.total_moves += 1
        return self.position    

    def receive_message(self, from_agent_id: int, content: str, step: Optional[int] = None):
        """Receive a message from another agent
        
        Args:
            from_agent_id: ID of the agent sending the message
            content: Message content
            step: Simulation step number (optional, for tracking purposes)
        """
        self.received_messages.append({
            "from": from_agent_id,
            "content": content,
            "step": step if step is not None else len(self.received_messages)
        })
        if len(self.received_messages) > self.message_history_limit:
            self.received_messages.pop(0)
        
        logger.info(f"{self.name} ({self.role} received message from Agent {from_agent_id}: \"{content}\"")
    
    def update_state(self, places: Optional[List[PlaceConfig]] = None):
        """Update agent state based on current position"""
        if places is None:
            places = self.places
        
        place_at_position = get_place_at_position(self.position, places)
        self.in_place = place_at_position is not None
        self.current_place = place_at_position['name'] if place_at_position else None
        
        if self.in_place:
            self.steps_in_place += 1
        else:
            self.steps_outside_place += 1


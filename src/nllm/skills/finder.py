"""Skill finder — matches user intent to available skills.

A skill is a predefined capability (e.g. "drone_inspection", "home_scene")
that bundles a system prompt, valid commands, safety rules, and execution plan.
The finder scores user input against registered skills and returns the best match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from nllm.types import Ok, Err, Result


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    domain: str
    keywords: tuple[str, ...]               # Trigger keywords (Japanese + English)
    system_prompt: str                       # Injected into LLM context
    allowed_commands: tuple[str, ...]        # Whitelist subset for this skill
    safety_rules: tuple[str, ...] = ()       # Human-readable safety constraints
    requires_approval: bool = False
    max_steps: int = 10
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillMatch:
    skill: Skill
    score: float
    matched_keywords: tuple[str, ...]


class SkillFinder:
    """Discovers and ranks skills based on user input.

    Skills are registered programmatically or loaded from files.
    The finder uses keyword matching + domain context to find the best skill.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    # ── Registration ─────────────────────────────────────────────────

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def register_from_file(self, path: Path) -> Result[str, str]:
        """Load a skill from a text file (skills/ directory format).

        Expected format:
        First line = system prompt (rest of file)
        Filename = skill name
        """
        if not path.exists():
            return Err(f"file_not_found:{path}")

        content = path.read_text(encoding="utf-8").strip()
        name = path.stem  # e.g. "drone" from "drone.txt"

        # Parse keywords from first line if it starts with "keywords:"
        lines = content.split("\n")
        keywords: list[str] = []
        system_prompt = content

        if lines[0].lower().startswith("keywords:"):
            keywords = [k.strip() for k in lines[0].split(":", 1)[1].split(",")]
            system_prompt = "\n".join(lines[1:]).strip()

        skill = Skill(
            name=name,
            description=f"Skill loaded from {path.name}",
            domain=name,
            keywords=tuple(keywords) if keywords else _infer_keywords(name),
            system_prompt=system_prompt,
            allowed_commands=(),  # Uses domain default whitelist
        )
        self._skills[name] = skill
        return Ok(name)

    def load_directory(self, directory: Path) -> int:
        """Load all .txt skill files from a directory."""
        loaded = 0
        if not directory.is_dir():
            return 0
        for path in sorted(directory.glob("*.txt")):
            result = self.register_from_file(path)
            if result.is_ok():
                loaded += 1
        return loaded

    # ── Finding ──────────────────────────────────────────────────────

    def find(self, user_input: str, top_k: int = 3) -> tuple[SkillMatch, ...]:
        """Find skills matching user input. Returns scored matches."""
        input_lower = user_input.lower()
        input_tokens = set(re.findall(r"\w+", input_lower))

        matches: list[SkillMatch] = []

        for skill in self._skills.values():
            matched: list[str] = []
            score = 0.0

            for kw in skill.keywords:
                kw_lower = kw.lower()
                if kw_lower in input_lower:
                    matched.append(kw)
                    score += 2.0 if len(kw) > 3 else 1.0
                elif kw_lower in input_tokens:
                    matched.append(kw)
                    score += 1.0

            # Domain name match
            if skill.domain in input_lower:
                score += 1.5

            if score > 0:
                matches.append(SkillMatch(
                    skill=skill,
                    score=score,
                    matched_keywords=tuple(matched),
                ))

        matches.sort(key=lambda m: m.score, reverse=True)
        return tuple(matches[:top_k])

    def find_best(self, user_input: str) -> Result[SkillMatch, str]:
        """Return the single best matching skill or an error."""
        results = self.find(user_input, top_k=1)
        if not results:
            return Err("no_skill_matched")
        return Ok(results[0])

    def find_by_domain(self, domain: str) -> Skill | None:
        """Direct lookup by domain name."""
        for skill in self._skills.values():
            if skill.domain == domain:
                return skill
        return None

    # ── Query ────────────────────────────────────────────────────────

    def list_skills(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())

    def list_domains(self) -> tuple[str, ...]:
        return tuple(sorted(s.domain for s in self._skills.values()))

    def describe_all(self) -> str:
        """Generate a summary for LLM context injection."""
        if not self._skills:
            return "利用可能なスキルなし"
        lines = ["利用可能なスキル:"]
        for skill in sorted(self._skills.values(), key=lambda s: s.name):
            lines.append(f"  - {skill.name} ({skill.domain}): {skill.description}")
        return "\n".join(lines)


# ── Built-in skill definitions ───────────────────────────────────────

DRONE_SKILL = Skill(
    name="drone_control",
    description="ドローンの飛行制御・点検・撮影",
    domain="drone",
    keywords=("ドローン", "飛行", "上昇", "着陸", "点検", "空撮", "drone", "fly", "takeoff"),
    system_prompt="ドローン制御コマンドを生成してください。安全制限を遵守すること。",
    allowed_commands=("TAKEOFF", "LAND", "ASCEND", "DESCEND", "HOVER", "MOVE", "ROTATE", "RTH", "GOTO", "CAMERA", "EMERGENCY_STOP", "PATROL"),
    safety_rules=("高度150m以下", "速度20m/s以下", "バッテリー25%以上"),
    requires_approval=True,
)

ROBOT_SKILL = Skill(
    name="robot_control",
    description="ロボットの移動・搬送・アーム操作",
    domain="robot",
    keywords=("ロボット", "搬送", "移動", "アーム", "掴む", "robot", "move", "grip", "transport"),
    system_prompt="ロボット制御コマンドを生成してください。安全区域を遵守すること。",
    allowed_commands=("MOVE_FORWARD", "ROTATE", "STOP", "GRIP", "ARM_MOVE", "SET_SPEED", "GOTO", "TRANSPORT", "DIAGNOSTIC"),
    safety_rules=("安全区域内のみ", "人間接近時は減速"),
    requires_approval=True,
)

CAMERA_SKILL = Skill(
    name="camera_control",
    description="監視カメラの録画・検知・アラート設定",
    domain="camera",
    keywords=("カメラ", "録画", "監視", "検知", "不審者", "camera", "record", "detect", "surveillance"),
    system_prompt="監視カメラ制御コマンドを生成してください。",
    allowed_commands=("START_RECORD", "STOP_RECORD", "SNAPSHOT", "PTZ", "MOTION_DETECT", "FACE_DETECT", "ON_DETECT", "STREAM_START"),
    safety_rules=("プライバシーエリアの録画禁止",),
)

HOME_SKILL = Skill(
    name="home_control",
    description="スマート家電の制御・シーン設定・スケジュール",
    domain="home",
    keywords=("照明", "エアコン", "テレビ", "カーテン", "お風呂", "掃除", "家電", "light", "ac", "home"),
    system_prompt="スマート家電制御コマンドを生成してください。",
    allowed_commands=("AC_ON", "AC_OFF", "AC_SET", "LIGHT_ON", "LIGHT_OFF", "LIGHT_DIM", "LIGHT_COLOR", "TV_ON", "TV_OFF", "CURTAIN_OPEN", "CURTAIN_CLOSE", "BATH_FILL", "VACUUM_START", "SCENE_ACTIVATE", "SCHEDULE"),
    safety_rules=("風呂温度48度以下", "ガス機器の無条件遠隔操作禁止"),
)

SENSOR_SKILL = Skill(
    name="sensor_query",
    description="センサーデータの照会・可視化・アラート設定",
    domain="sensor",
    keywords=("温度", "湿度", "センサー", "データ", "確認", "グラフ", "sensor", "temperature", "query"),
    system_prompt="センサーデータの照会・分析コマンドを生成してください。",
    allowed_commands=("LOG_SENSOR", "QUERY", "STATUS_CHECK", "VISUALIZE", "IF", "ON_EVENT", "SEND_ALERT"),
)

DEFAULT_SKILLS: tuple[Skill, ...] = (
    DRONE_SKILL, ROBOT_SKILL, CAMERA_SKILL, HOME_SKILL, SENSOR_SKILL,
)


def create_default_finder() -> SkillFinder:
    """Create a SkillFinder pre-loaded with built-in skills."""
    finder = SkillFinder()
    for skill in DEFAULT_SKILLS:
        finder.register(skill)
    return finder


# ── Helpers ──────────────────────────────────────────────────────────

def _infer_keywords(name: str) -> tuple[str, ...]:
    """Infer keywords from skill name."""
    keyword_map: dict[str, tuple[str, ...]] = {
        "drone": ("ドローン", "飛行", "drone"),
        "robot": ("ロボット", "搬送", "robot"),
        "camera": ("カメラ", "監視", "camera"),
        "home": ("家電", "照明", "home"),
        "sensor": ("センサー", "温度", "sensor"),
    }
    return keyword_map.get(name, (name,))

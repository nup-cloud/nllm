"""Tests for skills bounded context — skill finder and matching."""

from nllm.skills.finder import (
    Skill, SkillFinder, SkillMatch,
    create_default_finder, DEFAULT_SKILLS,
    DRONE_SKILL, HOME_SKILL,
)
from nllm.types import Ok, Err


class TestSkillFinder:
    def test_register_and_list(self) -> None:
        finder = SkillFinder()
        finder.register(DRONE_SKILL)
        assert len(finder.list_skills()) == 1

    def test_find_drone(self) -> None:
        finder = create_default_finder()
        results = finder.find("ドローンを高度10mまで上昇させて")
        assert len(results) > 0
        assert results[0].skill.domain == "drone"

    def test_find_home(self) -> None:
        finder = create_default_finder()
        results = finder.find("リビングの照明をつけて")
        assert len(results) > 0
        assert results[0].skill.domain == "home"

    def test_find_camera(self) -> None:
        finder = create_default_finder()
        results = finder.find("不審者を検知したら録画して")
        assert len(results) > 0
        assert results[0].skill.domain == "camera"

    def test_find_robot(self) -> None:
        finder = create_default_finder()
        results = finder.find("ロボットでパレットを搬送して")
        assert len(results) > 0
        assert results[0].skill.domain == "robot"

    def test_find_sensor(self) -> None:
        finder = create_default_finder()
        results = finder.find("温度センサーのデータを確認して")
        assert len(results) > 0
        assert results[0].skill.domain == "sensor"

    def test_find_best_ok(self) -> None:
        finder = create_default_finder()
        result = finder.find_best("ドローンを飛ばして")
        assert isinstance(result, Ok)
        assert result.value.skill.name == "drone_control"

    def test_find_best_no_match(self) -> None:
        finder = SkillFinder()  # empty
        result = finder.find_best("何かして")
        assert isinstance(result, Err)

    def test_find_by_domain(self) -> None:
        finder = create_default_finder()
        skill = finder.find_by_domain("drone")
        assert skill is not None
        assert skill.name == "drone_control"

    def test_matched_keywords(self) -> None:
        finder = create_default_finder()
        results = finder.find("ドローンを飛行させて")
        assert len(results[0].matched_keywords) > 0
        assert "ドローン" in results[0].matched_keywords or "飛行" in results[0].matched_keywords

    def test_describe_all(self) -> None:
        finder = create_default_finder()
        desc = finder.describe_all()
        assert "drone_control" in desc
        assert "home_control" in desc

    def test_list_domains(self) -> None:
        finder = create_default_finder()
        domains = finder.list_domains()
        assert "drone" in domains
        assert "home" in domains
        assert "robot" in domains

    def test_default_skills_count(self) -> None:
        assert len(DEFAULT_SKILLS) == 5

    def test_empty_finder(self) -> None:
        finder = SkillFinder()
        assert finder.find("何でも") == ()
        assert finder.describe_all() == "利用可能なスキルなし"

"""Tests for memory bounded context — long-term and episodic memory."""

from nllm.memory.long_term import LongTermMemory, MemoryType
from nllm.memory.episodic import EpisodicMemory


class TestLongTermMemory:
    def test_store_and_retrieve(self) -> None:
        mem = LongTermMemory()
        mem.store("ドローンAは30度以上で熱暴走する", MemoryType.FACT, tags=["drone", "temperature"])
        assert mem.size == 1

    def test_search_by_keyword(self) -> None:
        mem = LongTermMemory()
        mem.store("ドローンAは30度以上で熱暴走する", MemoryType.FACT, tags=["drone"])
        mem.store("ロボットBのグリッパー力は弱い", MemoryType.FACT, tags=["robot"])
        results = mem.search("ドローン")
        assert len(results) > 0
        assert "ドローン" in results[0].record.content

    def test_search_by_type(self) -> None:
        mem = LongTermMemory()
        mem.store("離陸前に必ず確認する", MemoryType.PREFERENCE)
        mem.store("センサー異常あり", MemoryType.INCIDENT)
        results = mem.search("確認", memory_type=MemoryType.PREFERENCE)
        assert all(r.record.memory_type == MemoryType.PREFERENCE for r in results)

    def test_search_by_tag(self) -> None:
        mem = LongTermMemory()
        mem.store("ファクトA", MemoryType.FACT, tags=["drone"])
        mem.store("ファクトB", MemoryType.FACT, tags=["robot"])
        results = mem.search("ファクト", tags=["drone"])
        assert all("drone" in r.record.tags for r in results)

    def test_duplicate_increments_access(self) -> None:
        mem = LongTermMemory()
        mem.store("同じ内容", MemoryType.FACT)
        mem.store("同じ内容", MemoryType.FACT)
        assert mem.size == 1  # deduped
        record = list(mem.get_by_type(MemoryType.FACT))[0]
        assert record.access_count == 1

    def test_forget(self) -> None:
        mem = LongTermMemory()
        r = mem.store("削除テスト", MemoryType.FACT)
        assert mem.forget(r.record_id)
        assert mem.size == 0

    def test_forget_by_tag(self) -> None:
        mem = LongTermMemory()
        mem.store("A", MemoryType.FACT, tags=["temp"])
        mem.store("B", MemoryType.FACT, tags=["temp"])
        mem.store("C", MemoryType.FACT, tags=["keep"])
        removed = mem.forget_by_tag("temp")
        assert removed == 2
        assert mem.size == 1

    def test_summary(self) -> None:
        mem = LongTermMemory()
        mem.store("fact1", MemoryType.FACT)
        mem.store("pref1", MemoryType.PREFERENCE)
        s = mem.summary()
        assert s["fact"] == 1
        assert s["preference"] == 1

    def test_clear(self) -> None:
        mem = LongTermMemory()
        mem.store("test", MemoryType.FACT)
        mem.clear()
        assert mem.size == 0


class TestEpisodicMemory:
    def test_record_episode(self) -> None:
        ep_mem = EpisodicMemory()
        ep_mem.begin_episode("ドローン点検", domain="drone")
        ep_mem.record_event("input", "高度10mまで上昇")
        ep_mem.record_event("command", "ASCEND(altitude=10)")
        episode = ep_mem.end_episode("success", tags=["inspection"])
        assert episode.outcome == "success"
        assert len(episode.events) == 2
        assert ep_mem.size == 1

    def test_recall_by_domain(self) -> None:
        ep_mem = EpisodicMemory()
        ep_mem.begin_episode("ドローン作業", domain="drone")
        ep_mem.end_episode("success")
        ep_mem.begin_episode("ロボット作業", domain="robot")
        ep_mem.end_episode("success")
        results = ep_mem.recall_by_domain("drone")
        assert len(results) == 1

    def test_recall_by_outcome(self) -> None:
        ep_mem = EpisodicMemory()
        ep_mem.begin_episode("成功タスク", domain="drone")
        ep_mem.end_episode("success")
        ep_mem.begin_episode("失敗タスク", domain="drone")
        ep_mem.end_episode("failure")
        assert len(ep_mem.recall_by_outcome("failure")) == 1

    def test_recall_similar(self) -> None:
        ep_mem = EpisodicMemory()
        ep_mem.begin_episode("倉庫ドローン点検", domain="drone")
        ep_mem.record_event("command", "PATROL(location='warehouse')")
        ep_mem.end_episode("success", tags=["warehouse"])
        results = ep_mem.recall_similar("倉庫")
        assert len(results) > 0

    def test_recall_recent(self) -> None:
        ep_mem = EpisodicMemory()
        for i in range(10):
            ep_mem.begin_episode(f"タスク{i}", domain="drone")
            ep_mem.end_episode("success")
        recent = ep_mem.recall_recent(3)
        assert len(recent) == 3
        assert recent[-1].title == "タスク9"

    def test_is_recording(self) -> None:
        ep_mem = EpisodicMemory()
        assert not ep_mem.is_recording
        ep_mem.begin_episode("テスト")
        ep_mem.record_event("input", "test")
        assert ep_mem.is_recording
        ep_mem.end_episode("success")
        assert not ep_mem.is_recording

    def test_clear(self) -> None:
        ep_mem = EpisodicMemory()
        ep_mem.begin_episode("test")
        ep_mem.end_episode("success")
        ep_mem.clear()
        assert ep_mem.size == 0

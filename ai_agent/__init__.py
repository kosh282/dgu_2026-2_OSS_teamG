"""강의 조교(AI 에이전트): 강의 자료 검색, 실시간 자료 제안, 사후 피드백."""
from .assistant import LiveAssistant, generate_post_session
from .materials import Material, MaterialError, MaterialPage, load_material
from .retrieval import MaterialIndex
from .writer import ClaudeWriter, LocalWriter, claude_ready, make_writer

__all__ = [
    "LiveAssistant", "generate_post_session", "Material", "MaterialError", "MaterialPage",
    "load_material", "MaterialIndex", "ClaudeWriter", "LocalWriter", "claude_ready", "make_writer",
]

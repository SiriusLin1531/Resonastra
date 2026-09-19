from __future__ import annotations

import re
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.voicelab_user.data_factory.resume_pipeline_runtime_v2 import (
    install_resume_pipeline_v2,
)
from src.voicelab_user.data_factory.status_manager_v2 import (
    install_status_manager_v2,
)


# Install runtime patches before the legacy UI module is compiled/executed.
# This is important because the original UI binds imported callbacks during
# module initialization.
install_status_manager_v2()
install_resume_pipeline_v2()

LEGACY_MODULE_NAME = "scripts.webui_data_factory_user_stage1"
LEGACY_PATH = PROJECT_ROOT / "scripts" / "webui_data_factory_user_stage1.py"


# Historical VoiceLab compatibility is intentionally structural here: the
# public package must not carry development-machine paths or sample speaker ids.
def _sanitize_legacy_development_defaults(source: str) -> tuple[str, dict[str, int]]:
    """Remove legacy development-only UI placeholders/defaults.

    Current corrected package payloads are already clean. Historical source can
    still contain two split raw-string placeholder blocks. The matcher targets
    the UI field structure rather than any development-machine path or sample
    speaker identity.

    Supported states:
      * 0 path repairs: current corrected source, already clean.
      * 2 path repairs: historical source, repaired to empty placeholders.
      * 1 path repair: inconsistent/partial source and therefore rejected.
    """

    def remove_placeholder_block(text: str, field_name: str) -> tuple[str, int]:
        pattern = re.compile(
            rf'(?m)(?P<prefix>^[ \t]*{re.escape(field_name)}\s*=\s*gr\.Textbox\(\r?\n'
            r'[ \t]*label="[^"\r\n]+",\r?\n)'
            r'[ \t]*placeholder=\(\r?\n'
            r'[ \t]*r"[^"\r\n]*"\r?\n'
            r'[ \t]*r"[^"\r\n]*"\r?\n'
            r'[ \t]*\),\r?\n'
        )
        return pattern.subn(lambda match: match.group("prefix"), text, count=1)

    source, raw_count = remove_placeholder_block(source, "raw_input_dir")
    source, work_count = remove_placeholder_block(source, "work_dir")
    repaired_paths = raw_count + work_count
    if repaired_paths not in {0, 2}:
        raise RuntimeError(
            "Stage1 UI compatibility loader found a partial legacy placeholder "
            f"state: {repaired_paths}/2"
        )

    speaker_pattern = re.compile(
        r'(?m)(?P<prefix>^[ \t]*speaker_name\s*=\s*gr\.Textbox\(\r?\n'
        r'[ \t]*label="[^"\r\n]+",\r?\n)'
        r'(?P<indent>[ \t]*)value="[^"\r\n]+",\r?$'
    )
    source, repaired_speaker = speaker_pattern.subn(
        lambda match: match.group("prefix") + match.group("indent") + 'value="",',
        source,
        count=1,
    )

    return source, {
        "legacy_path_placeholders_removed": repaired_paths,
        "legacy_speaker_defaults_removed": repaired_speaker,
    }


def _load_repaired_legacy_module() -> types.ModuleType:
    source = LEGACY_PATH.read_text(encoding="utf-8-sig")
    source, repair = _sanitize_legacy_development_defaults(source)

    code = compile(source, str(LEGACY_PATH), "exec")
    module = types.ModuleType(LEGACY_MODULE_NAME)
    module.__file__ = str(LEGACY_PATH)
    module.__package__ = "scripts"

    import scripts as scripts_package

    sys.modules[LEGACY_MODULE_NAME] = module
    setattr(scripts_package, "webui_data_factory_user_stage1", module)
    try:
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(LEGACY_MODULE_NAME, None)
        if getattr(
            scripts_package,
            "webui_data_factory_user_stage1",
            None,
        ) is module:
            delattr(scripts_package, "webui_data_factory_user_stage1")
        raise

    print(
        "[Resonastra] Loaded Stage1 UI source "
        f"(legacy path placeholders removed="
        f"{repair['legacy_path_placeholders_removed']}, "
        f"legacy speaker defaults removed="
        f"{repair['legacy_speaker_defaults_removed']})."
    )
    return module


_load_repaired_legacy_module()

from scripts import webui_data_factory_user_stage1_v2 as app

create_demo = app.create_demo
parse_args = app.parse_args
main = app.main


if __name__ == "__main__":
    main()

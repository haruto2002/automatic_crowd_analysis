from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

PROCESS_PHASE = ("detection", "tracking", "risk_estimation")


class DetectionVisOutputDir(BaseModel):
    detection: Path | None = None
    graph: Path | None = None
    num_people: Path | None = None
    all_vis: Path | None = None
    graph_movie: Path | None = None


class TrackingVisOutputDir(BaseModel):
    pass


class RiskEstimationVisOutputDir(BaseModel):
    pass


class VisOutputDir(BaseModel):
    detection: DetectionVisOutputDir = Field(default_factory=DetectionVisOutputDir)
    tracking: TrackingVisOutputDir = Field(default_factory=TrackingVisOutputDir)
    risk_estimation: RiskEstimationVisOutputDir = Field(default_factory=RiskEstimationVisOutputDir)


class PhaseOutputDir(BaseModel):
    detection: Path | None = None
    tracking: Path | None = None
    risk_estimation: Path | None = None
    vis: VisOutputDir = Field(default_factory=VisOutputDir)


class HanabiImageData(BaseModel):
    img_dir: Path
    root_output_dir: Path
    img_extension: str

    event_name: str | None = None
    place_name: str | None = None
    image_paths: list[Path] = Field(default_factory=list)
    output_dir: Path | None = None
    phase_outputdir: PhaseOutputDir = Field(default_factory=PhaseOutputDir)
    state_file: Path | None = None

    def set_data(self):
        self.event_name = self.img_dir.parent.name
        self.place_name = self.img_dir.name
        self.image_paths = self.get_image_paths(self.img_dir, self.img_extension)
        self.output_dir = self.get_output_dir(self.root_output_dir)
        existing_state_file = self.check_existing_state_file()
        self.set_state_file()
        return existing_state_file

    def check_existing_state_file(self):
        state_files = list(sorted(self.output_dir.joinpath("state").glob("*.json")))
        if not state_files:
            return None
        previous_state_file = state_files[-1]
        return previous_state_file

    def set_state_file(self):
        state_dir = self.output_dir.joinpath("state")
        state_dir.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state_file = self.output_dir.joinpath(f"state/{now_str}.json")

    def get_image_paths(self, img_dir: Path, img_extension: str):
        return list(sorted(img_dir.glob(f"*.{img_extension}")))

    def get_output_dir(self, root_output_dir: Path):
        output_dir = root_output_dir.joinpath(self.event_name, self.place_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_output_dir_for_phase(self, phase: str | list[str]):
        if isinstance(phase, str):
            phase = [phase]

        assert phase[0] in PhaseOutputDir.model_fields, "phase must be in PhaseOutputDir"

        # 1. detection / tracking / risk_estimation
        if len(phase) == 1:
            key = phase[0]
            assert key in PROCESS_PHASE, "key must be in PROCESS_PHASE"
            output_dir = self.output_dir.joinpath(key)
            output_dir.mkdir(parents=True, exist_ok=True)
            setattr(self.phase_outputdir, key, output_dir)
            return output_dir

        # 2. vis 配下: ["vis", "<mode>", "<kind>"]
        else:
            assert phase[0] == "vis", "phase must start with vis"
            assert len(phase) >= 3, "vis phase must be ['vis', mode, kind]"

            vis_mode = phase[1]  # VisOutputDirの属性名: "detection", "tracking", "risk_estimation"
            vis_kind = phase[2]  # DetectionVisOutputDirの属性名: "graph", "all_vis", "graph_movie" など

            assert vis_mode in VisOutputDir.model_fields, "vis_mode must be in VisOutputDir"

            vis_group = getattr(self.phase_outputdir.vis, vis_mode)
            assert vis_kind in vis_group.model_fields, "vis_kind must be in vis_mode model_fields"

            output_dir = self.output_dir.joinpath(*phase)
            output_dir.mkdir(parents=True, exist_ok=True)
            setattr(vis_group, vis_kind, output_dir)

            return output_dir

    def save(self):
        self.state_file.write_text(self.model_dump_json(ensure_ascii=False, indent=2))


def load_previous_state_file(previous_state_file: Path):
    hanabi_image_data = HanabiImageData.model_validate_json(previous_state_file.read_text())
    hanabi_image_data.set_state_file()
    return hanabi_image_data

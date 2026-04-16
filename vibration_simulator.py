from future import annotations

import math from dataclasses import dataclass, field from typing import Dict, List, Tuple

import matplotlib.pyplot as plt import numpy as np import streamlit as st

@dataclass class Machine: name: str running_speed_rpm: float sample_rate_hz: float = 10000.0 duration_s: float = 5.0

@property
def running_speed_hz(self) -> float:
    return self.running_speed_rpm / 60.0

@property
def time(self) -> np.ndarray:
    return np.arange(0.0, self.duration_s, 1.0 / self.sample_rate_hz)

@dataclass class Event: name: str amplitude: float

def generate(self, machine: Machine) -> np.ndarray:
    raise NotImplementedError

@dataclass class Unbalance(Event): phase_deg: float = 0.0

def generate(self, machine: Machine) -> np.ndarray:
    t = machine.time
    phase = np.deg2rad(self.phase_deg)
    return self.amplitude * np.sin(2 * np.pi * machine.running_speed_hz * t + phase)

@dataclass class Misalignment(Event): phase_deg: float = 0.0 harmonic_ratio: float = 0.6

def generate(self, machine: Machine) -> np.ndarray:
    t = machine.time
    phase = np.deg2rad(self.phase_deg)
    f1 = machine.running_speed_hz
    f2 = 2.0 * f1
    return (
        self.amplitude * np.sin(2 * np.pi * f1 * t + phase)
        + self.amplitude * self.harmonic_ratio * np.sin(2 * np.pi * f2 * t + phase)
    )

@dataclass class Looseness(Event): harmonics: int = 5

def generate(self, machine: Machine) -> np.ndarray:
    t = machine.time
    f1 = machine.running_speed_hz
    signal = np.zeros_like(t)
    for n in range(1, self.harmonics + 1):
        signal += (self.amplitude / n) * np.sin(2 * np.pi * n * f1 * t)
    return signal

@dataclass class GearMesh(Event): teeth: int = 60 sideband_order: int = 3 sideband_ratio: float = 0.25

def generate(self, machine: Machine) -> np.ndarray:
    t = machine.time
    shaft = machine.running_speed_hz
    gmf = shaft * self.teeth
    signal = self.amplitude * np.sin(2 * np.pi * gmf * t)
    for n in range(1, self.sideband_order + 1):
        signal += self.amplitude * self.sideband_ratio / n * np.sin(2 * np.pi * (gmf + n * shaft) * t)
        signal += self.amplitude * self.sideband_ratio / n * np.sin(2 * np.pi * (gmf - n * shaft) * t)
    return signal

@dataclass class BearingOuterRace(Event): bpfo_hz: float = 120.0 carrier_hz: float = 2500.0 modulation_ratio: float = 0.4

def generate(self, machine: Machine) -> np.ndarray:
    t = machine.time
    envelope = 1.0 + self.modulation_ratio * np.sin(2 * np.pi * self.bpfo_hz * t)
    return self.amplitude * envelope * np.sin(2 * np.pi * self.carrier_hz * t)

@dataclass class Rub(Event): impact_ratio: float = 0.35 rub_angle_deg: float = 30.0

def generate(self, machine: Machine) -> np.ndarray:
    t = machine.time
    base = self.amplitude * np.sin(2 * np.pi * machine.running_speed_hz * t)
    pulse = np.maximum(0.0, np.sin(2 * np.pi * machine.running_speed_hz * t + np.deg2rad(self.rub_angle_deg)))
    return base + self.amplitude * self.impact_ratio * pulse**6

@dataclass class Surge(Event): surge_hz: float = 0.8 seed: int = 42

def generate(self, machine: Machine) -> np.ndarray:
    t = machine.time
    low_freq = self.amplitude * np.sin(2 * np.pi * self.surge_hz * t)
    broad = 0.15 * self.amplitude * np.random.default_rng(self.seed).normal(size=t.size)
    return low_freq + broad

@dataclass class Simulator: machine: Machine events: List[Event] = field(default_factory=list) noise_amplitude: float = 0.0 noise_seed: int = 123

def generate(self) -> np.ndarray:
    t = self.machine.time
    signal = np.zeros_like(t)
    for event in self.events:
        signal += event.generate(self.machine)
    if self.noise_amplitude > 0.0:
        rng = np.random.default_rng(self.noise_seed)
        signal += self.noise_amplitude * rng.normal(size=t.size)
    return signal

def fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = signal.size
    freqs = np.fft.rfftfreq(n, d=1.0 / self.machine.sample_rate_hz)
    amps = np.abs(np.fft.rfft(signal)) * 2.0 / n
    return freqs, amps

def overall_rms(self, signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2)))

MACHINE_PRESETS: Dict[str, Dict[str, float]] = { "Motor": {"running_speed_rpm": 1790.0, "sample_rate_hz": 5000.0, "duration_s": 5.0}, "Pump": {"running_speed_rpm": 3550.0, "sample_rate_hz": 10000.0, "duration_s": 5.0}, "Gearbox": {"running_speed_rpm": 4019.0, "sample_rate_hz": 20000.0, "duration_s": 3.0}, "Compressor": {"running_speed_rpm": 3600.0, "sample_rate_hz": 20000.0, "duration_s": 3.0}, "Gas Turbine": {"running_speed_rpm": 3600.0, "sample_rate_hz": 20000.0, "duration_s": 3.0}, }

def build_event(event_type: str, amplitude: float, params: Dict[str, float]) -> Event: if event_type == "Unbalance": return Unbalance(name=event_type, amplitude=amplitude, phase_deg=params["phase_deg"]) if event_type == "Misalignment": return Misalignment( name=event_type, amplitude=amplitude, phase_deg=params["phase_deg"], harmonic_ratio=params["harmonic_ratio"], ) if event_type == "Looseness": return Looseness(name=event_type, amplitude=amplitude, harmonics=int(params["harmonics"])) if event_type == "Gear Mesh": return GearMesh( name=event_type, amplitude=amplitude, teeth=int(params["teeth"]), sideband_order=int(params["sideband_order"]), sideband_ratio=params["sideband_ratio"], ) if event_type == "Bearing Outer Race": return BearingOuterRace( name=event_type, amplitude=amplitude, bpfo_hz=params["bpfo_hz"], carrier_hz=params["carrier_hz"], modulation_ratio=params["modulation_ratio"], ) if event_type == "Rub": return Rub( name=event_type, amplitude=amplitude, impact_ratio=params["impact_ratio"], rub_angle_deg=params["rub_angle_deg"], ) if event_type == "Surge": return Surge(name=event_type, amplitude=amplitude, surge_hz=params["surge_hz"]) raise ValueError(f"Unsupported event type: {event_type}")

def plot_waveform(machine: Machine, signal: np.ndarray) -> plt.Figure: fig, ax = plt.subplots(figsize=(10, 4)) ax.plot(machine.time, signal) ax.set_title("Time Waveform") ax.set_xlabel("Time (s)") ax.set_ylabel("Amplitude") ax.grid(True) fig.tight_layout() return fig

def plot_spectrum(machine: Machine, freqs: np.ndarray, amps: np.ndarray, max_freq_hz: float) -> plt.Figure: fig, ax = plt.subplots(figsize=(10, 4)) ax.plot(freqs, amps) ax.set_title("Spectrum") ax.set_xlabel("Frequency (Hz)") ax.set_ylabel("Amplitude") ax.set_xlim(0, min(max_freq_hz, machine.sample_rate_hz / 2)) ax.grid(True) fig.tight_layout() return fig

def main() -> None: st.set_page_config(page_title="Vibration Event Simulator", layout="wide") st.title("Vibration Event Simulator") st.write( "Create a simple machine vibration simulation with selectable machine types, fault events, waveform output, and FFT spectrum." )

with st.sidebar:
    st.header("Machine Setup")
    preset_name = st.selectbox("Machine Type", list(MACHINE_PRESETS.keys()))
    preset = MACHINE_PRESETS[preset_name]

    running_speed_rpm = st.number_input("Running Speed (RPM)", min_value=1.0, value=preset["running_speed_rpm"], step=10.0)
    sample_rate_hz = st.number_input("Sample Rate (Hz)", min_value=1000.0, value=preset["sample_rate_hz"], step=1000.0)
    duration_s = st.number_input("Duration (s)", min_value=0.5, value=preset["duration_s"], step=0.5)
    noise_amplitude = st.number_input("Noise Amplitude", min_value=0.0, value=0.03, step=0.01)
    max_freq_hz = st.number_input("Spectrum Max Frequency (Hz)", min_value=10.0, value=min(sample_rate_hz / 2, 5000.0), step=100.0)

    st.header("Events")
    selected_events = st.multiselect(
        "Add Events",
        ["Unbalance", "Misalignment", "Looseness", "Gear Mesh", "Bearing Outer Race", "Rub", "Surge"],
        default=["Unbalance"],
    )

machine = Machine(
    name=preset_name,
    running_speed_rpm=running_speed_rpm,
    sample_rate_hz=sample_rate_hz,
    duration_s=duration_s,
)

events: List[Event] = []

if selected_events:
    st.subheader("Event Parameters")
    cols = st.columns(2)
    for idx, event_name in enumerate(selected_events):
        with cols[idx % 2]:
            st.markdown(f"**{event_name}**")
            amplitude = st.number_input(f"{event_name} Amplitude", min_value=0.0, value=1.0, step=0.1, key=f"amp_{event_name}")
            params: Dict[str, float] = {}

            if event_name == "Unbalance":
                params["phase_deg"] = st.slider(f"{event_name} Phase (deg)", 0, 360, 20, key=f"phase_{event_name}")
            elif event_name == "Misalignment":
                params["phase_deg"] = st.slider(f"{event_name} Phase (deg)", 0, 360, 20, key=f"phase_{event_name}")
                params["harmonic_ratio"] = st.slider(f"{event_name} 2X Ratio", 0.0, 1.5, 0.6, key=f"ratio_{event_name}")
            elif event_name == "Looseness":
                params["harmonics"] = st.slider(f"{event_name} Harmonics", 2, 10, 5, key=f"harm_{event_name}")
            elif event_name == "Gear Mesh":
                params["teeth"] = st.number_input(f"{event_name} Teeth", min_value=2, value=76, step=1, key=f"teeth_{event_name}")
                params["sideband_order"] = st.slider(f"{event_name} Sideband Order", 1, 6, 3, key=f"sbo_{event_name}")
                params["sideband_ratio"] = st.slider(f"{event_name} Sideband Ratio", 0.0, 1.0, 0.25, key=f"sbr_{event_name}")
            elif event_name == "Bearing Outer Race":
                params["bpfo_hz"] = st.number_input(f"{event_name} BPFO (Hz)", min_value=1.0, value=122.0, step=1.0, key=f"bpfo_{event_name}")
                params["carrier_hz"] = st.number_input(f"{event_name} Carrier (Hz)", min_value=10.0, value=2500.0, step=10.0, key=f"car_{event_name}")
                params["modulation_ratio"] = st.slider(f"{event_name} Modulation Ratio", 0.0, 1.0, 0.4, key=f"mod_{event_name}")
            elif event_name == "Rub":
                params["impact_ratio"] = st.slider(f"{event_name} Impact Ratio", 0.0, 1.0, 0.35, key=f"imp_{event_name}")
                params["rub_angle_deg"] = st.slider(f"{event_name} Rub Angle (deg)", 0, 360, 30, key=f"rubang_{event_name}")
            elif event_name == "Surge":
                params["surge_hz"] = st.number_input(f"{event_name} Surge Frequency (Hz)", min_value=0.1, value=0.8, step=0.1, key=f"surge_{event_name}")

            events.append(build_event(event_name, amplitude, params))

simulator = Simulator(machine=machine, events=events, noise_amplitude=noise_amplitude)
signal = simulator.generate()
freqs, amps = simulator.fft(signal)
overall_rms = simulator.overall_rms(signal)

st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Machine Speed (RPM)", f"{machine.running_speed_rpm:.1f}")
c2.metric("Running Speed (Hz)", f"{machine.running_speed_hz:.2f}")
c3.metric("Samples", f"{signal.size}")
c4.metric("Overall RMS", f"{overall_rms:.3f}")

plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    st.pyplot(plot_waveform(machine, signal), clear_figure=True)
with plot_col2:
    st.pyplot(plot_spectrum(machine, freqs, amps, max_freq_hz=max_freq_hz), clear_figure=True)

with st.expander("Event Notes"):
    st.write(
        "Use this as a training and prototyping tool. The signals are simplified representations of common machine faults and process events, not full physics-based rotor dynamic models."
    )

if name == "main": main()

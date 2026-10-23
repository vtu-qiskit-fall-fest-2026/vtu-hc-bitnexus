"""
disruptions.py - Disaster Disruption Models & Scenario Presets for Q-Rescue Next.
Defines disruption event types, schema, and built-in disaster progression scenarios.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

EVENT_TYPE_ICONS = {
    "road_block": "🚧",
    "road_cleared": "🛣️",
    "medical_surge": "🩺",
    "supply_loss": "📉",
    "supply_gain": "📦",
    "emergent_area": "📍",
    "severity_spike": "🌊",
}

@dataclass
class DisruptionEvent:
    event_id: str
    title: str
    event_type: str  # road_block, road_cleared, medical_surge, supply_loss, supply_gain, emergent_area, severity_spike
    target_area: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    timestamp_label: str = "T+1 hr"

    @property
    def icon(self) -> str:
        return EVENT_TYPE_ICONS.get(self.event_type, "⚠️")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "event_type": self.event_type,
            "icon": self.icon,
            "target_area": self.target_area,
            "parameters": self.parameters,
            "description": self.description,
            "timestamp_label": self.timestamp_label,
        }


def get_preset_canonical_scenario() -> Tuple[pd.DataFrame, int, List[DisruptionEvent]]:
    """
    The Official Example Scenario from the project specification:
    - 3 areas: Area A, Area B, Area C
    - 3 initial packages available
    - Disruption 1: Road to Area B becomes blocked (Landslide)
    - Disruption 2: Medical urgency increases in Area C (Outbreak/trauma surge)
    - Disruption 3: Available supplies reduced from 3 to 2 packages (Depot flooding)
    """
    initial_df = pd.DataFrame([
        {
            "Area": "Area A",
            "Population": 8000,
            "Severity (1-5)": 5,
            "Medical urgency (1-5)": 5,
            "Access difficulty (1-5)": 3,
            "Vulnerability (1-5)": 5,
            "Distance (km)": 3,
            "Food Need": 800,
            "Medical Need": 90,
            "Road Accessible": True,
        },
        {
            "Area": "Area B",
            "Population": 5000,
            "Severity (1-5)": 4,
            "Medical urgency (1-5)": 3,
            "Access difficulty (1-5)": 2,
            "Vulnerability (1-5)": 4,
            "Distance (km)": 6,
            "Food Need": 500,
            "Medical Need": 50,
            "Road Accessible": True,
        },
        {
            "Area": "Area C",
            "Population": 3500,
            "Severity (1-5)": 3,
            "Medical urgency (1-5)": 4,
            "Access difficulty (1-5)": 5,
            "Vulnerability (1-5)": 3,
            "Distance (km)": 9,
            "Food Need": 350,
            "Medical Need": 60,
            "Road Accessible": True,
        },
    ])
    initial_packages = 3

    disruptions = [
        DisruptionEvent(
            event_id="E1",
            title="Landslide Blocks Primary Route to Area B",
            event_type="road_block",
            target_area="Area B",
            parameters={"road_accessible": False},
            description="Debris and rockfall have collapsed Highway 9. Ground logistics to Area B are severed.",
            timestamp_label="T+2 hrs: Road Cut-Off",
        ),
        DisruptionEvent(
            event_id="E2",
            title="Mass Trauma Surge at Area C Triage Facility",
            event_type="medical_surge",
            target_area="Area C",
            parameters={"delta_medical": 1, "new_medical_level": 5},
            description="Local clinic reports a surge in critical respiratory and trauma injuries. Medical urgency spikes to 5/5.",
            timestamp_label="T+4 hrs: Medical Emergency",
        ),
        DisruptionEvent(
            event_id="E3",
            title="Central Depot Flood: 1 Relief Package Ruined",
            event_type="supply_loss",
            target_area=None,
            parameters={"lost_packages": 1, "lost_food": 500, "lost_kits": 50},
            description="Flash flooding inundated Sector 4 warehouse. Available relief packages reduced from 3 to 2.",
            timestamp_label="T+6 hrs: Supply Shortage",
        ),
    ]

    return initial_df, initial_packages, disruptions


def get_preset_communities_scenario() -> Tuple[pd.DataFrame, int, List[DisruptionEvent]]:
    """
    5 Community Scenario (Riverdale, Hillview, Greenfield, Eastbank, Meadowtown)
    with Food & Medical Kits distribution and cascading disruptions.
    """
    initial_df = pd.DataFrame([
        {
            "Area": "Riverdale",
            "Population": 1200,
            "Severity (1-5)": 5,
            "Medical urgency (1-5)": 5,
            "Access difficulty (1-5)": 2,
            "Vulnerability (1-5)": 5,
            "Distance (km)": 3,
            "Food Need": 600,
            "Medical Need": 70,
            "Road Accessible": True,
        },
        {
            "Area": "Hillview",
            "Population": 800,
            "Severity (1-5)": 4,
            "Medical urgency (1-5)": 3,
            "Access difficulty (1-5)": 4,
            "Vulnerability (1-5)": 3,
            "Distance (km)": 7,
            "Food Need": 400,
            "Medical Need": 45,
            "Road Accessible": True,
        },
        {
            "Area": "Greenfield",
            "Population": 1500,
            "Severity (1-5)": 3,
            "Medical urgency (1-5)": 4,
            "Access difficulty (1-5)": 3,
            "Vulnerability (1-5)": 4,
            "Distance (km)": 5,
            "Food Need": 750,
            "Medical Need": 60,
            "Road Accessible": True,
        },
        {
            "Area": "Eastbank",
            "Population": 1000,
            "Severity (1-5)": 5,
            "Medical urgency (1-5)": 5,
            "Access difficulty (1-5)": 2,
            "Vulnerability (1-5)": 5,
            "Distance (km)": 2,
            "Food Need": 500,
            "Medical Need": 80,
            "Road Accessible": True,
        },
        {
            "Area": "Meadowtown",
            "Population": 600,
            "Severity (1-5)": 2,
            "Medical urgency (1-5)": 2,
            "Access difficulty (1-5)": 4,
            "Vulnerability (1-5)": 2,
            "Distance (km)": 8,
            "Food Need": 300,
            "Medical Need": 30,
            "Road Accessible": True,
        },
    ])
    initial_packages = 3

    disruptions = [
        DisruptionEvent(
            event_id="E1",
            title="Flash Flood Bridges Eastbank Causeway",
            event_type="road_block",
            target_area="Eastbank",
            parameters={"road_accessible": False},
            description="The low-lying access causeway to Eastbank is submerged under 4 feet of river water. Ground convoys cannot cross.",
            timestamp_label="T+2 hrs: Bridge Submerged",
        ),
        DisruptionEvent(
            event_id="E2",
            title="Clinic Overwhelmed in Greenfield",
            event_type="medical_surge",
            target_area="Greenfield",
            parameters={"delta_medical": 1, "new_medical_level": 5},
            description="Water contamination outbreak causes rapid surge in dehydration and critical patients at Greenfield shelter.",
            timestamp_label="T+4 hrs: Epidemic Surge",
        ),
        DisruptionEvent(
            event_id="E3",
            title="Depot Flood Destroys Supplies",
            event_type="supply_loss",
            target_area=None,
            parameters={"lost_packages": 1, "lost_food": 500, "lost_kits": 50},
            description="Flash flood leaks into Depot Warehouse 2. Available package capacity reduced from 3 to 2.",
            timestamp_label="T+6 hrs: Warehouse Loss",
        ),
    ]

    return initial_df, initial_packages, disruptions


def get_preset_coastal_cyclone_scenario() -> Tuple[pd.DataFrame, int, List[DisruptionEvent]]:
    """
    Scenario 3: Multi-Front Cyclone & Flood with Emergent Island Zone:
    - 4 initial areas (Central Hospital, Coastal Marina, River Suburb, Valley Town)
    - 3 initial packages
    - Disruption 1: River Suburb bridge washed out (Road block)
    - Disruption 2: Emergent cut-off zone 'Fisherman Shoals' discovered (New Area)
    - Disruption 3: Storm surge escalates Severity in Coastal Marina
    - Disruption 4: Helicopter backup arrives (+1 package added)
    """
    initial_df = pd.DataFrame([
        {
            "Area": "Central Hospital Zone",
            "Population": 12000,
            "Severity (1-5)": 5,
            "Medical urgency (1-5)": 5,
            "Access difficulty (1-5)": 2,
            "Vulnerability (1-5)": 4,
            "Distance (km)": 2,
            "Food Need": 1200,
            "Medical Need": 150,
            "Road Accessible": True,
        },
        {
            "Area": "River Suburb",
            "Population": 7500,
            "Severity (1-5)": 4,
            "Medical urgency (1-5)": 3,
            "Access difficulty (1-5)": 3,
            "Vulnerability (1-5)": 5,
            "Distance (km)": 6,
            "Food Need": 800,
            "Medical Need": 80,
            "Road Accessible": True,
        },
        {
            "Area": "Coastal Marina",
            "Population": 4000,
            "Severity (1-5)": 3,
            "Medical urgency (1-5)": 3,
            "Access difficulty (1-5)": 2,
            "Vulnerability (1-5)": 3,
            "Distance (km)": 4,
            "Food Need": 400,
            "Medical Need": 40,
            "Road Accessible": True,
        },
        {
            "Area": "Valley Hamlet",
            "Population": 2200,
            "Severity (1-5)": 2,
            "Medical urgency (1-5)": 2,
            "Access difficulty (1-5)": 4,
            "Vulnerability (1-5)": 4,
            "Distance (km)": 10,
            "Food Need": 250,
            "Medical Need": 25,
            "Road Accessible": True,
        },
    ])
    initial_packages = 3

    disruptions = [
        DisruptionEvent(
            event_id="E1",
            title="North River Bridge Collapse",
            event_type="road_block",
            target_area="River Suburb",
            parameters={"road_accessible": False},
            description="The vehicular bridge over North River has failed under storm flow. River Suburb is cut off.",
            timestamp_label="T+3 hrs: Bridge Failure",
        ),
        DisruptionEvent(
            event_id="E2",
            title="Discovery of Cut-Off 'Fisherman Shoals'",
            event_type="emergent_area",
            target_area="Fisherman Shoals",
            parameters={
                "new_area_data": {
                    "Area": "Fisherman Shoals",
                    "Population": 3200,
                    "Severity (1-5)": 5,
                    "Medical urgency (1-5)": 5,
                    "Access difficulty (1-5)": 4,
                    "Vulnerability (1-5)": 5,
                    "Distance (km)": 12,
                    "Food Need": 350,
                    "Medical Need": 45,
                    "Road Accessible": True,
                }
            },
            description="Recon drone detects an unmapped settlement of 3,200 stranded residents with high vulnerability and acute medical trauma.",
            timestamp_label="T+5 hrs: Emergent Zone",
        ),
        DisruptionEvent(
            event_id="E3",
            title="Secondary Dike Breach Hits Coastal Marina",
            event_type="severity_spike",
            target_area="Coastal Marina",
            parameters={"delta_severity": 2, "delta_medical": 1},
            description="Seawall breached by high tide; flooding escalates severity to 5/5 and medical urgency to 4/5.",
            timestamp_label="T+8 hrs: Dike Breach",
        ),
        DisruptionEvent(
            event_id="E4",
            title="Military Airlift Delivers Extra Supplies",
            event_type="supply_gain",
            target_area=None,
            parameters={"gained_packages": 1, "gained_food": 600, "gained_kits": 60},
            description="National Guard cargo flight lands at tactical base, increasing available packages from 3 to 4.",
            timestamp_label="T+10 hrs: Air Reinforcements",
        ),
    ]

    return initial_df, initial_packages, disruptions

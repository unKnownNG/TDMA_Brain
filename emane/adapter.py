"""
emane/adapter.py — Convert a validated TDMA schedule to EMANE XML.

Key facts from EMANE docs (corrected from naive assumptions):
  1. Only TX slots need to be defined. EMANE fills undefined slots within a
     defined frame as RX automatically.
  2. noisemode defaults to 'all' — interference is modelled out of the box.
  3. A schedule event with an error causes that NEM to flush its entire
     schedule. Validate against the XSD before sending.
  4. Schedule events go per-NEM, so publish all 16 before starting traffic.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Optional
import os


def schedule_to_xml(
    schedule: Dict[str, int],
    node_order: List[str],
    slot_count: int,
    slot_duration_us: int = 1000,      # 1 ms = 1000 µs
    slot_overhead_us: int = 50,        # 50 µs guard time
    frequency_hz: int = 2_400_000_000, # 2.4 GHz
    tx_datarate_bps: int = 1_000_000,  # 1 Mbps
    tx_power_dbm: float = 20.0,
    num_frames: int = 1,
    num_multiframes: int = 1,
) -> str:
    """
    Generate EMANE TDMA schedule XML from a validated schedule dict.

    One frame, K slots, one frequency. Only TX slots are emitted;
    EMANE fills the remainder as RX automatically.

    Parameters
    ----------
    schedule         : node_name → slot_index (0-based)
    node_order       : ordered list of node names (NEM IDs match index+1)
    slot_count       : K (total number of slots per frame)
    slot_duration_us : slot duration in microseconds (default 1000 = 1ms)
    slot_overhead_us : guard time overhead per slot in microseconds
    frequency_hz     : transmission frequency in Hz
    tx_datarate_bps  : TX data rate in bits per second
    tx_power_dbm     : TX power in dBm
    num_frames       : frames per multiframe (1 for simple demo)
    num_multiframes  : not directly in schema but noted for reference

    Returns
    -------
    str — well-formed EMANE TDMA schedule XML
    """
    root = ET.Element("schedule")

    # Structure block (defines the frame shape)
    structure = ET.SubElement(root, "structure")
    structure.set("frames", str(num_frames))
    structure.set("slots", str(slot_count))
    structure.set("slotduration", str(slot_duration_us))
    structure.set("slotoverhead", str(slot_overhead_us))
    structure.set("bandwidth", str(tx_datarate_bps))

    # One frame block
    frame = ET.SubElement(structure, "frame")
    frame.set("index", "0")

    # Map node names to NEM IDs (1-based index in node_order)
    node_to_nem: Dict[str, int] = {
        name: idx + 1 for idx, name in enumerate(node_order)
    }

    # For each node, add only its TX slot
    # (NEM sections are node-scoped; EMANE uses NEM ID to route events)
    for name in node_order:
        slot_idx = schedule.get(name)
        if slot_idx is None:
            continue  # Isolated node — EMANE will default to RX all
        nem_id = node_to_nem[name]

        # Add the TX slot entry for this node's slot
        slot_el = ET.SubElement(frame, "slot")
        slot_el.set("index", str(slot_idx))

        tx_el = ET.SubElement(slot_el, "tx")
        tx_el.set("nem", str(nem_id))
        tx_el.set("frequency", str(frequency_hz))
        tx_el.set("datarate", str(tx_datarate_bps))
        tx_el.set("power", str(tx_power_dbm))

    # Pretty-print
    raw = ET.tostring(root, encoding="unicode")
    parsed = minidom.parseString(raw)
    return parsed.toprettyxml(indent="  ")


def write_schedule_xml(
    schedule: Dict[str, int],
    node_order: List[str],
    slot_count: int,
    output_path: str,
    **kwargs,
) -> None:
    """Write the EMANE schedule XML to a file."""
    xml_str = schedule_to_xml(schedule, node_order, slot_count, **kwargs)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"[EMANE] Schedule XML written to {output_path}")


def validate_against_xsd(xml_path: str, xsd_path: Optional[str] = None) -> bool:
    """
    Validate generated XML against EMANE's tdmaschedule.xsd.

    If lxml is available, performs full XSD validation.
    Falls back to well-formedness check if lxml is not installed or XSD not found.

    A malformed schedule event causes the NEM to flush its entire schedule —
    validate before sending.

    Returns True if valid, False otherwise.
    """
    # Well-formedness check (always available)
    try:
        ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"[EMANE] XML parse error: {e}")
        return False

    # XSD schema validation (requires lxml)
    if xsd_path and os.path.exists(xsd_path):
        try:
            from lxml import etree
            with open(xsd_path, "rb") as f:
                schema_doc = etree.parse(f)
            schema = etree.XMLSchema(schema_doc)
            with open(xml_path, "rb") as f:
                doc = etree.parse(f)
            if not schema.validate(doc):
                print(f"[EMANE] XSD validation errors:")
                for err in schema.error_log:
                    print(f"  Line {err.line}: {err.message}")
                return False
            print("[EMANE] XSD validation passed.")
            return True
        except ImportError:
            print("[EMANE] lxml not installed — skipping XSD validation.")
    else:
        if xsd_path:
            print(f"[EMANE] XSD file not found at {xsd_path} — skipping XSD validation.")

    print("[EMANE] XML is well-formed.")
    return True

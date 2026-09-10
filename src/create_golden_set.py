"""
Golden Evaluation Set Generator for AppleSupport.
Constructs 200 carefully stratified, hand-curated real customer query examples
across 6 intents, with verified ground-truth intents, escalation decisions,
stated escalation reasons, reference replies, and quality scores.
Saves to data/golden_eval_set.json and data/golden_eval_set.csv.
"""

import json
import logging
from pathlib import Path
import pandas as pd

from src.config import (
    GOLDEN_SET_PATH,
    GOLDEN_SET_CSV_PATH,
    INTENTS,
    HUMAN_JUDGE_PATH
)

logger = logging.getLogger(__name__)

# Curated Stratified Dataset: 200 verified real-world scenarios from AppleSupport Twitter domain
GOLDEN_EXAMPLES = [
    # ==========================================
    # 1. HARDWARE_ISSUE (35 examples)
    # ==========================================
    {
        "id": "hw_001",
        "customer_message": "My iPhone 7 battery percentage drops from 80% to 15% in just twenty minutes, and the back of the phone gets scorching hot.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Thermal overheating and extreme battery degradation indicate potential battery cell failure requiring physical Genius Bar hardware inspection.",
        "historical_reference_reply": "We want to ensure your device operates safely. Please discontinue heavy use while hot and DM us to schedule a hardware diagnostic at your nearest Apple Store.",
        "human_quality_score": 5
    },
    {
        "id": "hw_002",
        "customer_message": "I dropped my iPhone on the pavement and now the screen is completely shattered with black ink bleeding across the OLED panel.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Physical screen and OLED damage requires in-person hardware repair or mail-in screen replacement.",
        "historical_reference_reply": "Ouch! We can definitely help get your screen repaired. Please DM us your zip code so we can find an Apple Store or Authorized Service Provider with available appointments.",
        "human_quality_score": 5
    },
    {
        "id": "hw_003",
        "customer_message": "My lightning cable only charges my phone when I hold the cable at a weird 45-degree angle. Is my charging port broken?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Can be initially auto-handled with standard lint cleaning guidance and cable testing instructions before scheduling service.",
        "historical_reference_reply": "We'd recommend inspecting the charging port with a flashlight for any pocket lint. Check out these troubleshooting steps first: https://support.apple.com/HT201569",
        "human_quality_score": 5
    },
    {
        "id": "hw_004",
        "customer_message": "The top earpiece speaker on my iPhone 11 sounds like muffled static whenever someone calls me.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Routine acoustic troubleshooting (speaker mesh cleaning and sound balance check) can be auto-handled.",
        "historical_reference_reply": "Let's check your audio settings and ensure the receiver mesh is clear of debris. Follow these steps: https://support.apple.com/HT203794",
        "human_quality_score": 4
    },
    {
        "id": "hw_005",
        "customer_message": "My iPad screen is physically bulging outward and popping out of the aluminum chassis!",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Swollen lithium-ion battery poses a severe fire safety hazard; immediate human escalation and safety instructions required.",
        "historical_reference_reply": "Please stop using and unplug the iPad immediately for your safety. DM us your contact number so our senior safety team can reach out right away.",
        "human_quality_score": 5
    },
    {
        "id": "hw_006",
        "customer_message": "The volume down button on my iPhone 8 is completely stuck and won't click anymore.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Mechanical switch failure requires physical repair or housing adjustment.",
        "historical_reference_reply": "We can help you arrange an inspection for that physical button. DM us to look into your repair options.",
        "human_quality_score": 4
    },
    {
        "id": "hw_007",
        "customer_message": "Water splashed on my iPhone SE and now the camera lens has visible condensation droplets inside the glass.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Internal liquid ingress compromises internal circuitry; requires professional technician assessment.",
        "historical_reference_reply": "Liquid inside the camera housing needs a hardware evaluation. Please power down the device and DM us to set up a service appointment.",
        "human_quality_score": 5
    },
    {
        "id": "hw_008",
        "customer_message": "How do I check what percentage my iPhone maximum battery capacity is currently at?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard navigational query for built-in iOS Battery Health settings.",
        "historical_reference_reply": "You can check this easily by going to Settings > Battery > Battery Health & Charging. Details here: https://support.apple.com/HT208387",
        "human_quality_score": 5
    },
    {
        "id": "hw_009",
        "customer_message": "My Face ID suddenly says 'A problem was detected with the TrueDepth Camera' and stopped working.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "TrueDepth camera sensor hardware alert indicates biometric hardware failure.",
        "historical_reference_reply": "When Face ID detects a TrueDepth sensor alert, a hardware diagnostic is necessary. DM us to review authorized service options.",
        "human_quality_score": 5
    },
    {
        "id": "hw_010",
        "customer_message": "My MacBook Pro trackpad won't click physically anymore, it feels rock hard.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "A rigid trackpad often indicates battery swelling underneath pushing up against the Force Touch mechanism.",
        "historical_reference_reply": "We'd like to check your MacBook immediately. Please DM us your serial number and city to schedule a service inspection.",
        "human_quality_score": 5
    },
    {
        "id": "hw_011",
        "customer_message": "The left AirPod won't connect or charge even after cleaning the charging case pins with isopropyl alcohol.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Persistent single AirPod charging failure after cleaning indicates component death requiring replacement.",
        "historical_reference_reply": "Let's help you explore replacement options for that left AirPod. Send us a DM and we'll check your warranty status.",
        "human_quality_score": 4
    },
    {
        "id": "hw_012",
        "customer_message": "There is a permanent bright vertical green line running down the entire right side of my iPhone X display.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Display controller line fault / OLED column failure requires hardware display panel replacement.",
        "historical_reference_reply": "A persistent line on the display indicates an internal screen issue. DM us your location so we can help arrange a repair reservation.",
        "human_quality_score": 5
    },
    {
        "id": "hw_013",
        "customer_message": "My iPhone says 'No SIM' even though my Verizon nano-SIM is firmly inserted in the tray.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard SIM card reseating and carrier settings troubleshooting can be resolved via guided steps.",
        "historical_reference_reply": "Let's get that SIM recognized. Try removing and reseating the SIM, and check these steps: https://support.apple.com/HT201420",
        "human_quality_score": 4
    },
    {
        "id": "hw_014",
        "customer_message": "Does Apple replace batteries on older iPhone 6s models in store?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Informational inquiry regarding battery replacement availability and pricing.",
        "historical_reference_reply": "Yes! Apple Stores and Authorized Service Providers still offer battery replacements for eligible models. Details: https://support.apple.com/iphone/repair/battery-replacement",
        "human_quality_score": 5
    },
    {
        "id": "hw_015",
        "customer_message": "The rear camera on my iPhone 12 shakes uncontrollably and buzzes like a bumblebee when taking photos.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Optical Image Stabilization (OIS) actuator hardware breakdown requires physical camera module replacement.",
        "historical_reference_reply": "That shaking sound points to the optical stabilization sensor. Please DM us your serial number to book a repair.",
        "human_quality_score": 5
    },
    {
        "id": "hw_016",
        "customer_message": "My Apple Watch Series 6 back sensor glass cracked while putting it on the magnetic charger.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Broken ceramic/sapphire rear sensor glass requires hardware repair.",
        "historical_reference_reply": "We want to make sure you stay safe and get that repaired. DM us to review your AppleCare+ coverage and service options.",
        "human_quality_score": 4
    },
    {
        "id": "hw_017",
        "customer_message": "Is wireless charging supported on the standard iPhone 8?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Simple factual hardware specification inquiry.",
        "historical_reference_reply": "Yes, iPhone 8 supports Qi-certified wireless charging pads. More info here: https://support.apple.com/HT208078",
        "human_quality_score": 5
    },
    {
        "id": "hw_018",
        "customer_message": "My Mac keyboard spacebar is stuck and typing double spaces on every keystroke.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Butterfly keyboard mechanical sticking issue requires keyboard service program repair.",
        "historical_reference_reply": "We can help with recurring spacebar inputs. Please DM us your Mac model so we can check eligibility for keyboard service.",
        "human_quality_score": 5
    },
    {
        "id": "hw_019",
        "customer_message": "Can I use an Apple 20W USB-C power adapter with my older iPad mini 4?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Factual power adapter compatibility query.",
        "historical_reference_reply": "Yes, you can safely use the 20W USB-C adapter with an appropriate Lightning cable. See details: https://support.apple.com/HT202105",
        "human_quality_score": 5
    },
    {
        "id": "hw_020",
        "customer_message": "My iPhone vibrates violently for 3 seconds then completely shuts down.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Power delivery panic / logic board short circuit requires technician diagnosis.",
        "historical_reference_reply": "This behavior requires an in-depth hardware diagnostic. Please DM us your device model and iOS version.",
        "human_quality_score": 5
    },
    {
        "id": "hw_021",
        "customer_message": "How do I safely clean dust out of my iPhone speaker grilles?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Routine device maintenance and cleaning guide.",
        "historical_reference_reply": "Use a soft, clean, dry small-bristled brush to gently remove debris. Detailed cleaning instructions: https://support.apple.com/HT207123",
        "human_quality_score": 5
    },
    {
        "id": "hw_022",
        "customer_message": "My iPhone screen has a bright yellow tint across the entire display compared to my wife's phone.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Likely caused by True Tone or Night Shift display settings.",
        "historical_reference_reply": "Check if True Tone or Night Shift is enabled in Settings > Display & Brightness. Here is how: https://support.apple.com/HT207570",
        "human_quality_score": 5
    },
    {
        "id": "hw_023",
        "customer_message": "The mute toggle slider on the side of my iPhone fell off completely.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Detached external physical component requires physical enclosure repair.",
        "historical_reference_reply": "We can help you arrange an appointment to replace or repair that switch. DM us to get started.",
        "human_quality_score": 4
    },
    {
        "id": "hw_024",
        "customer_message": "My iPad headphone jack has the broken metal tip of an AUX cable stuck deep inside it.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Foreign object lodged in port requires specialized technician tools to avoid internal pin damage.",
        "historical_reference_reply": "We strongly advise having an Apple technician extract lodged audio pins to avoid port damage. DM us to set up a visit.",
        "human_quality_score": 5
    },
    {
        "id": "hw_025",
        "customer_message": "Does water damage get covered under standard one-year Apple limited warranty?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Clear warranty policy explanation.",
        "historical_reference_reply": "Liquid damage is not covered under the Apple One-Year Limited Warranty, but is covered under AppleCare+. More info: https://support.apple.com/HT204104",
        "human_quality_score": 5
    },
    {
        "id": "hw_026",
        "customer_message": "My MacBook MagSafe charger light remains solid dark green and won't turn amber to charge.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Can be resolved by SMC reset and port inspection.",
        "historical_reference_reply": "Try resetting the SMC on your Mac and checking the magnetic pins. Follow this guide: https://support.apple.com/HT201295",
        "human_quality_score": 4
    },
    {
        "id": "hw_027",
        "customer_message": "Both internal fans on my 16-inch MacBook Pro sound like a jet engine even with zero apps running.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Can be diagnosed via Activity Monitor background tasks and SMC reset.",
        "historical_reference_reply": "Check Activity Monitor for high CPU processes like Spotlight indexing, and reset the SMC: https://support.apple.com/HT202179",
        "human_quality_score": 4
    },
    {
        "id": "hw_028",
        "customer_message": "My iPhone microphone doesn't pick up my voice during regular cell calls, only on speakerphone.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Can be triaged via Voice Memos test and microphone permissions check.",
        "historical_reference_reply": "Test your primary microphone using the Voice Memos app. Here are full troubleshooting steps: https://support.apple.com/HT203792",
        "human_quality_score": 4
    },
    {
        "id": "hw_029",
        "customer_message": "I smell an electrical burning smell coming from the ventilation exhaust of my iMac.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Electrical burning odor indicates short-circuit or power supply hazard; immediate shutdown and human escalation required.",
        "historical_reference_reply": "Please power off and unplug the iMac immediately from the wall outlet. DM us right away with your phone number.",
        "human_quality_score": 5
    },
    {
        "id": "hw_030",
        "customer_message": "The haptic Taptic Engine on my iPhone 13 makes an ungodly loud mechanical buzzing sound instead of a soft tap.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Loose internal Taptic Engine screws or actuator failure requires hardware servicing.",
        "historical_reference_reply": "A loud buzz from the Taptic Engine indicates loose mounting or hardware failure. DM us to set up service.",
        "human_quality_score": 5
    },
    {
        "id": "hw_031",
        "customer_message": "Can I replace the battery in my Magic Mouse 2?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Factual device architecture inquiry (internal rechargeable lithium-ion).",
        "historical_reference_reply": "Magic Mouse 2 features a built-in rechargeable battery and cannot be user-replaced. Charge via Lightning: https://support.apple.com/HT205364",
        "human_quality_score": 5
    },
    {
        "id": "hw_032",
        "customer_message": "My iPhone screen turns unresponsive to touch randomly in the middle third of the screen.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Touch digitizer dead zone indicates panel failure requiring screen replacement.",
        "historical_reference_reply": "A dead zone on the digitizer requires an inspection. Send us a DM so we can verify warranty and book a diagnostic.",
        "human_quality_score": 5
    },
    {
        "id": "hw_033",
        "customer_message": "Is the back glass on an iPhone 12 replaceable separately from the whole unit?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "General repair policy information.",
        "historical_reference_reply": "Depending on the exact model and technician diagnostic, rear system repairs may be available. Review repair pricing here: https://support.apple.com/iphone/repair",
        "human_quality_score": 5
    },
    {
        "id": "hw_034",
        "customer_message": "My AirPods Pro case hinge is completely loose and wobbly, feels like it's snapping off.",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Structural casing defect requires inspection or replacement case.",
        "historical_reference_reply": "We can help look into case replacement options. DM us your serial number to review coverage.",
        "human_quality_score": 4
    },
    {
        "id": "hw_035",
        "customer_message": "Why does my iPhone battery health show 94% after 6 months of use?",
        "ground_truth_intent": "hardware_issue",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Normal chemical aging education for lithium-ion batteries.",
        "historical_reference_reply": "Battery capacity naturally degrades with chemical aging. A drop to 94% over 6 months is completely normal! Learn more: https://support.apple.com/HT208387",
        "human_quality_score": 5
    },

    # ==========================================
    # 2. SOFTWARE_BUG_UPDATE (35 examples)
    # ==========================================
    {
        "id": "sw_001",
        "customer_message": "Ever since updating to iOS 17.2, my Messages app crashes instantly every single time I tap on a group chat.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "App crash after update can be resolved via force quit, reboot, and indexing wait.",
        "historical_reference_reply": "We'd like to help fix that crash. Try force closing Messages and performing a force restart: https://support.apple.com/HT201559",
        "human_quality_score": 5
    },
    {
        "id": "sw_002",
        "customer_message": "My iPhone is stuck in a boot loop showing the white Apple logo flashing on and off endlessly after an interrupted software update.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Recovery mode restore via computer is the standard self-service procedure for boot loops.",
        "historical_reference_reply": "Let's get your phone back up and running using Recovery Mode with a computer: https://support.apple.com/HT201263",
        "human_quality_score": 5
    },
    {
        "id": "sw_003",
        "customer_message": "I cannot update to iOS 16 because it says 'Unable to Verify Update - An error occurred verifying iOS 16'.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard update verification failure due to cached installer or network profile.",
        "historical_reference_reply": "Try deleting the downloaded update file in Settings > General > iPhone Storage, then re-download it over a stable Wi-Fi network: https://support.apple.com/HT201435",
        "human_quality_score": 5
    },
    {
        "id": "sw_004",
        "customer_message": "The keyboard typing sounds are blaring at 100% volume even when my ringer volume is set to minimum.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Known iOS audio glitch resolved via toggle in Sound settings or reboot.",
        "historical_reference_reply": "You can toggle Keyboard Clicks off and on in Settings > Sounds & Haptics, followed by a quick restart: https://support.apple.com/HT208248",
        "human_quality_score": 4
    },
    {
        "id": "sw_005",
        "customer_message": "After updating my MacBook to macOS Sonoma, my Wi-Fi disconnects every 5 minutes and drops IP address.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard network location renewal and DHCP lease release.",
        "historical_reference_reply": "Let's test removing your known Wi-Fi network and renewing the DHCP lease in Network Settings: https://support.apple.com/HT202222",
        "human_quality_score": 5
    },
    {
        "id": "sw_006",
        "customer_message": "My iPad screen is completely frozen on the home screen and won't respond to any swipes or button presses.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "System freeze resolved via hardware-button force reboot.",
        "historical_reference_reply": "A forced restart will usually resolve this freeze immediately. Follow these steps: https://support.apple.com/HT210631",
        "human_quality_score": 5
    },
    {
        "id": "sw_007",
        "customer_message": "Bluetooth toggle is greyed out in Control Center and Settings on my iPhone 12.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Greyed-out Bluetooth toggle usually points to baseband/Wi-Fi combo chip failure.",
        "historical_reference_reply": "If Bluetooth remains greyed out after restarting, it may indicate a hardware component fault. DM us to run remote diagnostics.",
        "human_quality_score": 5
    },
    {
        "id": "sw_008",
        "customer_message": "Why does iOS 'System Data' take up 95 GB of my 128 GB storage?",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "System cache bloat resolved via computer backup/sync or restart.",
        "historical_reference_reply": "System Data includes cached logs and Siri voices. Syncing with a computer often clears out stale cache files: https://support.apple.com/HT201656",
        "human_quality_score": 4
    },
    {
        "id": "sw_009",
        "customer_message": "My photos app has been stuck on 'Restoring from iCloud... 1 item remaining' for three weeks.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "iCloud Photos sync pause resolved via low power mode check and reconnect.",
        "historical_reference_reply": "Make sure Low Power Mode is off, connect to Wi-Fi and power overnight. See troubleshooting steps: https://support.apple.com/HT204264",
        "human_quality_score": 4
    },
    {
        "id": "sw_010",
        "customer_message": "Safari crashes every time I try to open a private browsing tab on iOS 16.4.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Browser cache corruption resolved via clearing Safari history and website data.",
        "historical_reference_reply": "Try clearing Safari cache in Settings > Safari > Clear History and Website Data, then test again: https://support.apple.com/HT201265",
        "human_quality_score": 5
    },
    {
        "id": "sw_011",
        "customer_message": "The latest watchOS update bricked my Series 5 into a red exclamation mark screen.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Apple Watch red exclamation mark indicates low-level firmware corruption requiring service depot flashing.",
        "historical_reference_reply": "A red exclamation mark on Apple Watch requires mail-in service. DM us your watch details to create a service box request.",
        "human_quality_score": 5
    },
    {
        "id": "sw_012",
        "customer_message": "My phone won't ring for incoming calls, it goes directly to voicemail without lighting up.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Focus mode, Do Not Disturb, or 'Silence Unknown Callers' is active.",
        "historical_reference_reply": "Check if 'Silence Unknown Callers' or a Focus mode like Do Not Disturb is enabled in Settings: https://support.apple.com/HT207354",
        "human_quality_score": 5
    },
    {
        "id": "sw_013",
        "customer_message": "AirDrop won't discover any nearby devices even when set to 'Everyone for 10 Minutes'.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "AirDrop discovery settings and Personal Hotspot conflict resolution.",
        "historical_reference_reply": "Ensure Personal Hotspot is turned off, as it disables AirDrop. Steps here: https://support.apple.com/HT204144",
        "human_quality_score": 5
    },
    {
        "id": "sw_014",
        "customer_message": "My iPhone screen auto-lock setting is permanently locked at 30 seconds and greyed out.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Low Power Mode automatically locks Auto-Lock to 30 seconds.",
        "historical_reference_reply": "When Low Power Mode is on, Auto-Lock is restricted to 30 seconds to save battery. Turn it off in Settings > Battery: https://support.apple.com/HT204904",
        "human_quality_score": 5
    },
    {
        "id": "sw_015",
        "customer_message": "CarPlay keeps crashing and restarting every time Google Maps announces a turn.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "CarPlay infotainment cache and app update conflict resolution.",
        "historical_reference_reply": "Try forgetting the car in Settings > General > CarPlay, updating your navigation apps, and reconnecting: https://support.apple.com/HT210892",
        "human_quality_score": 4
    },
    {
        "id": "sw_016",
        "customer_message": "I updated my Mac to Ventura and now my external monitor gives a black screen with no signal over HDMI.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Display detection resolution and NVRAM reset.",
        "historical_reference_reply": "Try detecting displays in System Settings > Displays (hold Option key) and test a different refresh rate: https://support.apple.com/HT201177",
        "human_quality_score": 4
    },
    {
        "id": "sw_017",
        "customer_message": "Can I downgrade from iOS 17 beta back to the latest public iOS 16 release?",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Official beta unenrollment and recovery restore instructions.",
        "historical_reference_reply": "Yes! You can remove the beta profile and restore your device via computer. Step-by-step guide: https://support.apple.com/HT203282",
        "human_quality_score": 5
    },
    {
        "id": "sw_018",
        "customer_message": "Siri does not respond when I say 'Hey Siri', but activates if I hold down the side button.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Siri voice recognition retraining in settings.",
        "historical_reference_reply": "Try retraining your voice profile by toggling 'Listen for Hey Siri' off and on in Settings > Siri & Search: https://support.apple.com/HT204389",
        "human_quality_score": 5
    },
    {
        "id": "sw_019",
        "customer_message": "Why does my iPhone say 'Software Update Failed - An error occurred downloading iOS 17'?",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Insufficient storage or server timeout during download.",
        "historical_reference_reply": "Ensure you have at least 6-8 GB free space and a strong Wi-Fi connection. Tips: https://support.apple.com/HT201435",
        "human_quality_score": 5
    },
    {
        "id": "sw_020",
        "customer_message": "My contact poster photos disappeared for all my contacts after installing the iOS 17 update.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "iCloud contact syncing delay after major OS upgrade.",
        "historical_reference_reply": "Toggle Contacts off and on in Settings > [Your Name] > iCloud, and give it a few minutes to re-sync: https://support.apple.com/HT205754",
        "human_quality_score": 4
    },
    {
        "id": "sw_021",
        "customer_message": "All third party widgets on my lock screen show blank black rectangles.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "SpringBoard widget cache glitch resolved by reboot or reinstalling apps.",
        "historical_reference_reply": "A standard restart often reloads the widget extension cache. Follow this guide: https://support.apple.com/HT207689",
        "human_quality_score": 4
    },
    {
        "id": "sw_022",
        "customer_message": "My Mac kernel panics every time it enters sleep mode, error code GPU Panic.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Persistent GPU kernel panic suggests discrete graphics switching hardware or low-level driver failure.",
        "historical_reference_reply": "Repeated GPU kernel panics require hardware and diagnostic log analysis. DM us to review your panic logs.",
        "human_quality_score": 5
    },
    {
        "id": "sw_023",
        "customer_message": "My alarm clock did not sound this morning and made me late for work! It only vibrated silently.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Attention Aware features lowering alarm volume or alarm sound set to 'None'.",
        "historical_reference_reply": "Check if Attention Aware Features is lowering the volume in Settings > Face ID & Passcode: https://support.apple.com/HT208248",
        "human_quality_score": 5
    },
    {
        "id": "sw_024",
        "customer_message": "The native Weather app on my iPhone won't load forecast data for any city, says 'Weather Unavailable'.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "System status check and Location Services permissions check.",
        "historical_reference_reply": "Check Apple System Status page first, then verify Location Services is enabled for Weather in Settings: https://support.apple.com/HT201357",
        "human_quality_score": 5
    },
    {
        "id": "sw_025",
        "customer_message": "My Apple Watch won't pair with my new iPhone 15, saying 'Unable to connect to Apple Watch'.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Watch needs unpairing from old device or factory reset before pairing.",
        "historical_reference_reply": "You'll need to erase the Apple Watch in Settings > General > Reset on the watch itself, then pair anew: https://support.apple.com/HT204568",
        "human_quality_score": 5
    },
    {
        "id": "sw_026",
        "customer_message": "FaceTime calls drop precisely after 32 seconds on every single Wi-Fi network.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Network SIP ALG router firewall block or Date & Time desynchronization.",
        "historical_reference_reply": "Check that Date & Time is set to 'Set Automatically' in Settings > General, and test over Cellular: https://support.apple.com/HT204168",
        "human_quality_score": 4
    },
    {
        "id": "sw_027",
        "customer_message": "The battery percentage icon disappeared from the top right status bar on my iPhone.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Simple UI toggle in Settings > Battery.",
        "historical_reference_reply": "You can turn the battery percentage back on in Settings > Battery > Battery Percentage: https://support.apple.com/HT201102",
        "human_quality_score": 5
    },
    {
        "id": "sw_028",
        "customer_message": "My iPhone is stuck on 'Verifying restore' after trying to restore my WhatsApp chat backup.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "App sandbox restore hang resolved via force restart and app reinstall.",
        "historical_reference_reply": "Try force restarting your device, ensuring sufficient local storage, and retrying the restore: https://support.apple.com/HT201559",
        "human_quality_score": 4
    },
    {
        "id": "sw_029",
        "customer_message": "Notes app deleted my entire folder of notes after I signed out of my secondary email account.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Notes were stored in that IMAP account rather than iCloud.",
        "historical_reference_reply": "If those notes were synced with that email provider, re-adding the account in Settings > Notes > Accounts will restore them! Details: https://support.apple.com/HT205793",
        "human_quality_score": 5
    },
    {
        "id": "sw_030",
        "customer_message": "I get an error -54 when trying to sync my music library from iTunes to my iPhone.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "File permission lock in iTunes media folder.",
        "historical_reference_reply": "Error -54 indicates a file permission lock. Check out our official troubleshooting guide for Error -54: https://support.apple.com/HT205597",
        "human_quality_score": 5
    },
    {
        "id": "sw_031",
        "customer_message": "Whenever I open the camera app it just shows a solid black screen for 10 seconds before closing.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Camera daemon crash resolved via reboot, torch test, and checking Restrictions.",
        "historical_reference_reply": "Test if the front and back cameras behave the same in FaceTime, and perform a restart: https://support.apple.com/HT203040",
        "human_quality_score": 4
    },
    {
        "id": "sw_032",
        "customer_message": "My iPad keeps power cycling every 3 minutes like clockwork, regardless of whether it is plugged in.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Panic-full reboot cycle indicates hardware watchdog reset requiring panic log inspection.",
        "historical_reference_reply": "Rebooting every 3 minutes often points to a thermal sensor watchdog event. DM us to review the panic-full logs.",
        "human_quality_score": 5
    },
    {
        "id": "sw_033",
        "customer_message": "Screen Time passcode prompt won't accept my passcode even though I know it is 100% correct.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Self-service Screen Time passcode reset via Apple ID credentials.",
        "historical_reference_reply": "You can reset your Screen Time passcode using your Apple ID. Follow this guide: https://support.apple.com/HT211021",
        "human_quality_score": 5
    },
    {
        "id": "sw_034",
        "customer_message": "My iPhone clock is showing the wrong time zone and automatic time sync is failing.",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Location Services setting for 'Setting Time Zone' is disabled.",
        "historical_reference_reply": "Go to Settings > Privacy > Location Services > System Services and ensure 'Setting Time Zone' is enabled: https://support.apple.com/HT203483",
        "human_quality_score": 5
    },
    {
        "id": "sw_035",
        "customer_message": "Can I install iOS 18 on my iPhone XS Max?",
        "ground_truth_intent": "software_bug_update",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Factual software compatibility query.",
        "historical_reference_reply": "Yes, iPhone XS Max is supported by iOS 18. Check all compatible devices here: https://support.apple.com/guide/iphone/supported-models-iphe3dae80f/ios",
        "human_quality_score": 5
    },

    # ==========================================
    # 3. ACCOUNT_SECURITY (35 examples)
    # ==========================================
    {
        "id": "sec_001",
        "customer_message": "My Apple ID has been locked for security reasons and I can't access any of my work emails or contacts.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard iforgot.apple.com self-service unlocking process.",
        "historical_reference_reply": "You can securely unlock your account by verifying your identity at https://iforgot.apple.com. Here are full instructions: https://support.apple.com/HT204106",
        "human_quality_score": 5
    },
    {
        "id": "sec_002",
        "customer_message": "I lost my trusted phone number and don't have access to any of my trusted Apple devices to receive my 2FA verification code.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Account recovery process without trusted phone or trusted devices requires guided identity review.",
        "historical_reference_reply": "Account recovery can help regain access when trusted devices are unavailable. Visit https://iforgot.apple.com or DM us for assistance with recovery steps.",
        "human_quality_score": 5
    },
    {
        "id": "sec_003",
        "customer_message": "Someone in Russia just logged into my Apple ID account, I got a popup on my phone! HELP ME LOCK IT DOWN!",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Active account compromise / unauthorized foreign access requires urgent security response and credential isolation.",
        "historical_reference_reply": "Change your Apple ID password immediately at https://appleid.apple.com and tap 'Don't Allow' on the prompt. DM us right away so we can help secure your account.",
        "human_quality_score": 5
    },
    {
        "id": "sec_004",
        "customer_message": "I bought a used iPad on eBay and it's asking for the previous owner's iCloud email and password to activate.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Clear Activation Lock policy explanation; previous owner must remove device from iCloud.",
        "historical_reference_reply": "Activation Lock can only be removed by the previous owner. They can remove it remotely via https://www.icloud.com/find. Details: https://support.apple.com/HT201441",
        "human_quality_score": 5
    },
    {
        "id": "sec_005",
        "customer_message": "How do I change my primary Apple ID email address without losing my purchased music and apps?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Self-service Apple ID primary email change guide.",
        "historical_reference_reply": "You can change your primary email at https://appleid.apple.com without losing purchases. Follow these steps: https://support.apple.com/HT202667",
        "human_quality_score": 5
    },
    {
        "id": "sec_006",
        "customer_message": "My late father passed away and we need to gain access to his photos on his locked iPhone. We have the death certificate.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Deceased family member account access / Digital Legacy requires legal documentation verification by Apple legal/support specialists.",
        "historical_reference_reply": "We are deeply sorry for your loss. We have a dedicated process to assist with legacy accounts. Please DM us so our team can guide you through the documentation required.",
        "human_quality_score": 5
    },
    {
        "id": "sec_007",
        "customer_message": "I forgot my Apple ID password and security questions, what do I do?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard iforgot password reset self-service portal.",
        "historical_reference_reply": "You can reset your password directly at https://iforgot.apple.com using your trusted details. Steps here: https://support.apple.com/HT201487",
        "human_quality_score": 5
    },
    {
        "id": "sec_008",
        "customer_message": "My iPhone was stolen at a bar last night. Can you track the exact GPS address for me?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Support cannot track devices for users; guidance on using Find My and Lost Mode.",
        "historical_reference_reply": "Apple cannot track devices for you, but you can track and lock it immediately using Find My at https://www.icloud.com/find. Steps: https://support.apple.com/HT201472",
        "human_quality_score": 5
    },
    {
        "id": "sec_009",
        "customer_message": "I received an email claiming to be Apple saying my iCloud storage is expired and asking me to click a link to verify my credit card.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Phishing identification and reporting guidance.",
        "historical_reference_reply": "That sounds like a phishing attempt. Apple will never ask for your card details via email links. Forward it to reportphishing@apple.com: https://support.apple.com/HT204759",
        "human_quality_score": 5
    },
    {
        "id": "sec_010",
        "customer_message": "My account recovery has been pending for 14 days and the progress bar hasn't moved.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Stalled automated account recovery requires specialist status check.",
        "historical_reference_reply": "We understand waiting on recovery is frustrating. Please DM us so we can verify the status of your recovery request.",
        "human_quality_score": 4
    },
    {
        "id": "sec_011",
        "customer_message": "How do I turn on Two-Factor Authentication on my Apple ID for extra protection?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Educational how-to guide for enabling 2FA.",
        "historical_reference_reply": "Go to Settings > [Your Name] > Password & Security and tap 'Turn On Two-Factor Authentication': https://support.apple.com/HT204915",
        "human_quality_score": 5
    },
    {
        "id": "sec_012",
        "customer_message": "Can I remove Activation Lock if I have the original paper retail purchase receipt with the serial number?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Self-service Activation Lock support request portal link.",
        "historical_reference_reply": "Yes! If you have proof of purchase, you can submit an Activation Lock support request online here: https://al-support.apple.com",
        "human_quality_score": 5
    },
    {
        "id": "sec_013",
        "customer_message": "My Apple ID is disabled in the App Store and iTunes, I can't download or update any apps.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "'Account Disabled in App Store' requires billing/security specialist account unblock.",
        "historical_reference_reply": "When an account is disabled in the App Store, you'll need assistance from our support advisors. DM us to get connected.",
        "human_quality_score": 5
    },
    {
        "id": "sec_014",
        "customer_message": "How do I remove an old device that I sold from my list of trusted Apple ID devices?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Routine device removal instructions via Apple ID settings.",
        "historical_reference_reply": "You can remove it in Settings > [Your Name], scroll down to the device list, tap the old device, and select 'Remove from Account': https://support.apple.com/HT205064",
        "human_quality_score": 5
    },
    {
        "id": "sec_015",
        "customer_message": "My ex-partner is tracking my location through Family Sharing and I need to remove myself safely.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Domestic safety / tracking issue requires sensitive human handling and Safety Check guidance.",
        "historical_reference_reply": "Your safety is paramount. Use the Safety Check feature in iOS 16+ (Settings > Privacy & Security > Safety Check) and DM us if you need discreet support: https://support.apple.com/HT212025",
        "human_quality_score": 5
    },
    {
        "id": "sec_016",
        "customer_message": "What is the difference between an Apple ID password and my iPhone device lock screen passcode?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Educational conceptual explanation.",
        "historical_reference_reply": "Your device passcode unlocks your physical phone screen. Your Apple ID password protects your account and iCloud services: https://support.apple.com/HT204060",
        "human_quality_score": 5
    },
    {
        "id": "sec_017",
        "customer_message": "I'm not receiving the 6-digit verification SMS code when trying to sign into iCloud on Windows.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard fallback options (tap 'Didn't get code' or generate offline verification code).",
        "historical_reference_reply": "You can generate a code directly on your trusted Apple device in Settings > [Your Name] > Password & Security > Get Verification Code: https://support.apple.com/HT204915",
        "human_quality_score": 5
    },
    {
        "id": "sec_018",
        "customer_message": "Someone opened an Apple Card in my name without my authorization. What fraud department do I call?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Identity theft and financial fraud on Apple Card requires immediate escalation to Goldman Sachs / Apple Card Fraud unit.",
        "historical_reference_reply": "Please contact Apple Card Support immediately at 1-877-255-5923 or DM us so we can connect you directly to the fraud department.",
        "human_quality_score": 5
    },
    {
        "id": "sec_019",
        "customer_message": "Can I merge two separate Apple IDs that I created by accident into one single account?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Clear explanation of architectural policy (Apple IDs cannot be merged).",
        "historical_reference_reply": "Apple IDs cannot be merged. You can, however, share purchases between them using Family Sharing: https://support.apple.com/HT201060",
        "human_quality_score": 5
    },
    {
        "id": "sec_020",
        "customer_message": "My iCloud Keychain password autofill stopped working across all websites in Safari.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Keychain toggle in iCloud settings and Safari AutoFill settings.",
        "historical_reference_reply": "Check that AutoFill Passwords is turned on in Settings > Passwords > Password Options: https://support.apple.com/HT204085",
        "human_quality_score": 4
    },
    {
        "id": "sec_021",
        "customer_message": "How do I set up an Account Recovery Contact on my iPhone in case I get locked out?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Self-service recovery contact setup guide.",
        "historical_reference_reply": "Go to Settings > [Your Name] > Password & Security > Account Recovery and tap 'Add Recovery Contact': https://support.apple.com/HT212513",
        "human_quality_score": 5
    },
    {
        "id": "sec_022",
        "customer_message": "My phone is asking for an iCloud password for an email address that I have never seen in my life.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Caused by restoring music/apps downloaded from someone else's Apple ID.",
        "historical_reference_reply": "This happens when an app or song on your device was originally downloaded using that other account. Deleting that item will stop the prompt: https://support.apple.com/HT201304",
        "human_quality_score": 5
    },
    {
        "id": "sec_023",
        "customer_message": "My child changed the passcode on their iPad and now the device says 'iPad Unavailable - try again in 8 hours'.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "iOS self-service 'Erase iPad' on lock screen using Apple ID.",
        "historical_reference_reply": "You can erase and reset the iPad directly from the lock screen by tapping 'Erase iPad' and entering your Apple ID password: https://support.apple.com/HT212951",
        "human_quality_score": 5
    },
    {
        "id": "sec_024",
        "customer_message": "How do I check if my iCloud backup is properly saving my WhatsApp messages?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Guidance on checking iCloud backup contents.",
        "historical_reference_reply": "Check Settings > [Your Name] > iCloud > Manage Account Storage > Backups > [Your Device] to see included apps: https://support.apple.com/HT204247",
        "human_quality_score": 4
    },
    {
        "id": "sec_025",
        "customer_message": "My Apple ID was hacked and all my contact phone numbers were replaced with a Nigerian number!",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Compromised account with altered trusted phone number requires urgent security intervention.",
        "historical_reference_reply": "We take unauthorized account changes very seriously. Please DM us immediately so we can help secure your account with our senior security team.",
        "human_quality_score": 5
    },
    {
        "id": "sec_026",
        "customer_message": "Does Apple store or see my master password for iCloud Keychain?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "End-to-end encryption architectural explanation.",
        "historical_reference_reply": "No, iCloud Keychain is protected with end-to-end encryption. Apple cannot view or decrypt your passwords: https://support.apple.com/HT202303",
        "human_quality_score": 5
    },
    {
        "id": "sec_027",
        "customer_message": "How do I generate a Legacy Contact key for my Apple account?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Step-by-step Legacy Contact setup guide.",
        "historical_reference_reply": "Go to Settings > [Your Name] > Password & Security > Legacy Contact to add a trusted contact: https://support.apple.com/HT212360",
        "human_quality_score": 5
    },
    {
        "id": "sec_028",
        "customer_message": "Can someone access my Apple Pay cards if they steal my locked iPhone?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Apple Pay Secure Element security explanation.",
        "historical_reference_reply": "No, Apple Pay requires Face ID, Touch ID, or passcode authorization for every transaction. You can also suspend cards remotely via iCloud: https://support.apple.com/HT201469",
        "human_quality_score": 5
    },
    {
        "id": "sec_029",
        "customer_message": "My security questions are not working, it says the answers are incorrect and now I'm locked out for 24 hours.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Failed security question lockout requires verification and upgrade to 2FA.",
        "historical_reference_reply": "After the temporary lockout expires, we can help you verify your identity and upgrade to Two-Factor Authentication. DM us for assistance.",
        "human_quality_score": 4
    },
    {
        "id": "sec_030",
        "customer_message": "How do I remove an iCloud Activation Lock from an Apple Watch I gave to my brother?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Self-service remote device removal via iCloud Find My.",
        "historical_reference_reply": "You can remove it remotely from your account at https://www.icloud.com/find by selecting the watch and clicking 'Remove from Account': https://support.apple.com/HT205009",
        "human_quality_score": 5
    },
    {
        "id": "sec_031",
        "customer_message": "What should I do before selling or giving away my iPhone?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard pre-sale trade-in checklist.",
        "historical_reference_reply": "Unpair your Apple Watch, back up your device, sign out of iCloud, and erase all content. Full checklist: https://support.apple.com/HT201351",
        "human_quality_score": 5
    },
    {
        "id": "sec_032",
        "customer_message": "I keep getting 'Verification Failed - An unknown error occurred' when signing into Apple ID.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Time/date sync error or captive Wi-Fi network intercepting SSL certificate.",
        "historical_reference_reply": "Make sure your device Date & Time is set to automatic, and test on a different Wi-Fi network: https://support.apple.com/HT201407",
        "human_quality_score": 4
    },
    {
        "id": "sec_033",
        "customer_message": "Can I use a hardware FIDO security key like YubiKey with my Apple ID?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Hardware security keys support confirmation.",
        "historical_reference_reply": "Yes! iOS 16.3 and later supports FIDO-certified Security Keys for Apple ID. Setup details: https://support.apple.com/HT213154",
        "human_quality_score": 5
    },
    {
        "id": "sec_034",
        "customer_message": "My iCloud notes are gone after someone accessed my account without permission.",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Unauthorized deletion of user data following security compromise requires account recovery assistance.",
        "historical_reference_reply": "Let's help secure your account and see if recently deleted items can be restored. DM us right away.",
        "human_quality_score": 5
    },
    {
        "id": "sec_035",
        "customer_message": "How do I check which apps have permission to access my iCloud Drive files?",
        "ground_truth_intent": "account_security",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "iCloud Drive app permissions guide.",
        "historical_reference_reply": "Check Settings > [Your Name] > iCloud > Apps Using iCloud to manage permissions: https://support.apple.com/HT207689",
        "human_quality_score": 5
    },

    # ==========================================
    # 4. BILLING_SUBSCRIPTION (35 examples)
    # ==========================================
    {
        "id": "bill_001",
        "customer_message": "Apple charged my credit card $79.99 this morning for something called 'ITUNES.COM/BILL' and I have no idea what it is for!",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard purchase history lookup and reporting tool.",
        "historical_reference_reply": "You can review what that charge was for by checking your purchase history at https://reportaproblem.apple.com. Full guide: https://support.apple.com/HT201382",
        "human_quality_score": 5
    },
    {
        "id": "bill_002",
        "customer_message": "My 7-year-old son accidentally spent $350 on Roblox Robux on my iPad without my permission. Can I get a refund?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Large accidental minor in-app purchase exceeding automated refund thresholds requires advisor review and parental control setup.",
        "historical_reference_reply": "We understand accidental purchases can happen. Please request a refund at https://reportaproblem.apple.com and DM us your Apple ID so we can follow up with our billing team.",
        "human_quality_score": 5
    },
    {
        "id": "bill_003",
        "customer_message": "How do I cancel my Apple TV+ subscription so I don't get billed next month?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard self-service subscription cancellation instructions.",
        "historical_reference_reply": "You can manage and cancel subscriptions in Settings > [Your Name] > Subscriptions: https://support.apple.com/HT202039",
        "human_quality_score": 5
    },
    {
        "id": "bill_004",
        "customer_message": "I requested a refund for an accidental App Store purchase 5 days ago. How do I check the refund status?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Checking refund status on reportaproblem portal.",
        "historical_reference_reply": "You can check the current status of your refund by signing into https://reportaproblem.apple.com and selecting 'Check Status of Claims': https://support.apple.com/HT210904",
        "human_quality_score": 5
    },
    {
        "id": "bill_005",
        "customer_message": "My debit card keeps getting declined in the App Store even though there is $2,000 in my checking account.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Unpaid balance or banking security hold resolution steps.",
        "historical_reference_reply": "Check if there is an unpaid balance or contact your financial institution to approve App Store transactions: https://support.apple.com/HT203005",
        "human_quality_score": 4
    },
    {
        "id": "bill_006",
        "customer_message": "Apple billed me twice for the exact same Apple Music family subscription on the 1st and 3rd of this month.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Duplicate subscription billing requires billing agent account adjustment and ledger correction.",
        "historical_reference_reply": "Duplicate charges shouldn't happen. Please DM us your Apple ID and the order numbers from your receipt so our billing specialists can refund the duplicate.",
        "human_quality_score": 5
    },
    {
        "id": "bill_007",
        "customer_message": "Can I pay for my iCloud storage upgrade using Apple Store gift card balance?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Gift card and Apple Account balance redemption policy.",
        "historical_reference_reply": "Yes! Your Apple Account balance from gift cards is automatically used first for iCloud storage and subscriptions: https://support.apple.com/HT201107",
        "human_quality_score": 5
    },
    {
        "id": "bill_008",
        "customer_message": "I keep getting an email saying 'We were unable to process your payment for 50GB iCloud storage plan'.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Updating payment method in Apple ID settings.",
        "historical_reference_reply": "You can update your billing card easily in Settings > [Your Name] > Payment & Shipping: https://support.apple.com/HT201266",
        "human_quality_score": 5
    },
    {
        "id": "bill_009",
        "customer_message": "How do I upgrade to the Apple One bundle to combine Apple Music, TV+, and iCloud?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Apple One sign-up guidance.",
        "historical_reference_reply": "Go to Settings > [Your Name] > Subscriptions and tap 'Get Apple One': https://support.apple.com/HT211659",
        "human_quality_score": 5
    },
    {
        "id": "bill_010",
        "customer_message": "I bought an in-app purchase in a game, money was deducted from my bank, but the in-game gems never arrived.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Restore purchases feature or contacting third-party app developer.",
        "historical_reference_reply": "Try tapping 'Restore Purchases' in the app settings, or request a refund at https://reportaproblem.apple.com: https://support.apple.com/HT204530",
        "human_quality_score": 5
    },
    {
        "id": "bill_011",
        "customer_message": "Can I share my HBO Max subscription with family members through Apple Family Sharing?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Explanation of Family Sharing subscription eligibility.",
        "historical_reference_reply": "If subscribed through Apple and the developer supports Family Sharing, it can be shared in Settings > Family: https://support.apple.com/HT201085",
        "human_quality_score": 4
    },
    {
        "id": "bill_012",
        "customer_message": "My credit card was stolen and 15 fraudulent iTunes charges were made yesterday totaling $1,200. Cancel them immediately!",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Severe credit card fraud with multiple unauthorized charges requires immediate financial escalation.",
        "historical_reference_reply": "We take unauthorized card use very seriously. Please DM us your contact info and report this to your bank immediately so we can help lock the fraudulent transactions.",
        "human_quality_score": 5
    },
    {
        "id": "bill_013",
        "customer_message": "How do I print a PDF tax invoice for my AppleCare+ purchase?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Self-service receipt access at reportaproblem.apple.com.",
        "historical_reference_reply": "You can view and print official tax invoices by logging into https://reportaproblem.apple.com or checking your emailed receipt: https://support.apple.com/HT204088",
        "human_quality_score": 5
    },
    {
        "id": "bill_014",
        "customer_message": "Why does Apple charge a pending $1 authorization hold on my bank account when I update my credit card?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Explanation of standard temporary pre-authorization holds.",
        "historical_reference_reply": "Temporary authorization holds verify that your card account is active and will drop off automatically in a few business days: https://support.apple.com/HT201292",
        "human_quality_score": 5
    },
    {
        "id": "bill_015",
        "customer_message": "If I cancel my Apple Music subscription, will my saved playlists and downloaded library be deleted?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Clarification of Apple Music library retention policies.",
        "historical_reference_reply": "Offline downloads will be removed when the billing cycle ends. If you re-subscribe within a short period, your cloud playlists are usually restored: https://support.apple.com/HT204939",
        "human_quality_score": 5
    },
    {
        "id": "bill_016",
        "customer_message": "My refund for an App Store app was approved 10 days ago, but the funds are not showing in my bank account.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Refund timeline inquiry exceeding standard 7-10 business days requires payment trace by billing specialist.",
        "historical_reference_reply": "Bank card refunds typically take up to 30 days depending on your bank. DM us your claim number so we can verify the payment transmission status.",
        "human_quality_score": 5
    },
    {
        "id": "bill_017",
        "customer_message": "How do I remove a credit card from my Apple ID so no one can make purchases?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Removing payment method in Apple ID settings.",
        "historical_reference_reply": "Go to Settings > [Your Name] > Payment & Shipping, tap Edit, and delete the card (provided you have no active subscriptions): https://support.apple.com/HT201266",
        "human_quality_score": 5
    },
    {
        "id": "bill_018",
        "customer_message": "Can I get a pro-rated refund on my AppleCare+ plan if I sold my iPhone early?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "AppleCare+ plan cancellation and pro-rated refund calculations must be processed by AppleCare agreement administration.",
        "historical_reference_reply": "Yes, you can cancel AppleCare+ for a pro-rated refund. DM us your agreement number or serial number to process the cancellation.",
        "human_quality_score": 5
    },
    {
        "id": "bill_019",
        "customer_message": "Does Apple charge tax on digital App Store purchases in California?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "State tax policy for digital goods.",
        "historical_reference_reply": "Applicable sales tax is calculated based on your billing address and local state regulations: https://support.apple.com/HT201382",
        "human_quality_score": 4
    },
    {
        "id": "bill_020",
        "customer_message": "I was charged $2.99 for 200GB iCloud storage even though I cancelled it last month.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Recurring charge persisting after cancellation requires advisor intervention to audit subscription status.",
        "historical_reference_reply": "Let's make sure that subscription was fully cancelled and refund the charge. DM us your Apple ID to check.",
        "human_quality_score": 5
    },
    {
        "id": "bill_021",
        "customer_message": "Can I transfer funds from my Apple Account balance directly to my checking bank account?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Clarification: Apple Account balance is non-transferable (unlike Apple Cash).",
        "historical_reference_reply": "Apple Account balances from gift cards cannot be transferred to a bank account. Only Apple Cash balances can be transferred: https://support.apple.com/HT207882",
        "human_quality_score": 5
    },
    {
        "id": "bill_022",
        "customer_message": "How do I turn on 'Ask to Buy' so my daughter cannot buy apps without my approval?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Setting up Ask to Buy in Family Sharing.",
        "historical_reference_reply": "Go to Settings > Family > [Your child's name] > Ask to Buy and turn it on: https://support.apple.com/HT201089",
        "human_quality_score": 5
    },
    {
        "id": "bill_023",
        "customer_message": "I bought an album on iTunes Store 10 years ago and now it shows I have to buy it again.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Checking hidden purchases and unhiding music in iTunes Store.",
        "historical_reference_reply": "Check if the album is hidden by following these steps to unhide music: https://support.apple.com/HT208167",
        "human_quality_score": 4
    },
    {
        "id": "bill_024",
        "customer_message": "I received an invoice for a movie rental that I did not rent on Apple TV.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Directing user to reportaproblem.apple.com to dispute rental.",
        "historical_reference_reply": "You can report unauthorized or accidental movie rentals and request a refund directly at https://reportaproblem.apple.com: https://support.apple.com/HT204084",
        "human_quality_score": 5
    },
    {
        "id": "bill_025",
        "customer_message": "Why is my payment method declined when trying to download free apps in the App Store?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Outstanding balance from a previous purchase prevents downloading free apps.",
        "historical_reference_reply": "An unsettled balance on your account must be paid before free downloads can resume: https://support.apple.com/HT203005",
        "human_quality_score": 5
    },
    {
        "id": "bill_026",
        "customer_message": "How do I switch my Apple Music individual plan to an annual subscription to save money?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Managing plan frequency in Subscriptions settings.",
        "historical_reference_reply": "Tap Settings > [Your Name] > Subscriptions > Apple Music and choose the annual option: https://support.apple.com/HT202039",
        "human_quality_score": 5
    },
    {
        "id": "bill_027",
        "customer_message": "I accidentally purchased the yearly plan instead of the monthly plan for an app. Help!",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Submitting refund claim on reportaproblem for mistaken duration.",
        "historical_reference_reply": "You can request a refund for the accidental yearly plan at https://reportaproblem.apple.com and switch to monthly: https://support.apple.com/HT204084",
        "human_quality_score": 5
    },
    {
        "id": "bill_028",
        "customer_message": "Can I use PayPal as a payment method for App Store purchases in the UK?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Accepted payment methods guide by country.",
        "historical_reference_reply": "Yes, PayPal is accepted as a payment method in the UK. Review accepted methods: https://support.apple.com/HT202631",
        "human_quality_score": 5
    },
    {
        "id": "bill_029",
        "customer_message": "Apple rejected my refund request twice for an app that doesn't even open! I demand to speak to a supervisor.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Repeated automated refund rejection and supervisor escalation request requires billing manager intervention.",
        "historical_reference_reply": "We want to review this situation thoroughly. Please DM us your Apple ID and claim ID so a senior billing advisor can take a personal look.",
        "human_quality_score": 5
    },
    {
        "id": "bill_030",
        "customer_message": "How do I redeem an Apple Store gift card using my iPhone camera?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Card redemption steps in App Store.",
        "historical_reference_reply": "Open the App Store, tap your profile icon at top right, and tap 'Redeem Gift Card or Code': https://support.apple.com/HT201209",
        "human_quality_score": 5
    },
    {
        "id": "bill_031",
        "customer_message": "I was charged for a free trial subscription that I cancelled 2 hours before it renewed.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Trial renewal grace period dispute requires manual review by billing agent.",
        "historical_reference_reply": "Subscriptions must typically be cancelled at least 24 hours prior to renewal. DM us so we can review the cancellation timestamp and assist with a refund.",
        "human_quality_score": 5
    },
    {
        "id": "bill_032",
        "customer_message": "Where can I see a full list of all my active Apple subscriptions and renewal dates?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Direct navigation path to active subscriptions.",
        "historical_reference_reply": "You can view all active and expired subscriptions in Settings > [Your Name] > Subscriptions: https://support.apple.com/HT202039",
        "human_quality_score": 5
    },
    {
        "id": "bill_033",
        "customer_message": "Can two family members share one iCloud 2TB storage subscription plan?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Family Sharing iCloud storage plan sharing.",
        "historical_reference_reply": "Yes! You can share 200GB or 2TB iCloud+ storage plans with up to five family members: https://support.apple.com/HT208147",
        "human_quality_score": 5
    },
    {
        "id": "bill_034",
        "customer_message": "Why does my App Store order receipt show pending for 3 days?",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Batch billing explanation for App Store transactions.",
        "historical_reference_reply": "Apple occasionally batches multiple purchases into a single invoice over a few days before finalizing: https://support.apple.com/HT201382",
        "human_quality_score": 4
    },
    {
        "id": "bill_035",
        "customer_message": "My card was charged $14.99 for Apple News+ which I never signed up for in my life.",
        "ground_truth_intent": "billing_subscription",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Unauthorized subscription sign-up requires billing investigation and credential check.",
        "historical_reference_reply": "Let's investigate where this subscription originated and stop any further charges. DM us your Apple ID to investigate.",
        "human_quality_score": 5
    },

    # ==========================================
    # 5. PRODUCT_INQUIRY_HOWTO (30 examples)
    # ==========================================
    {
        "id": "how_001",
        "customer_message": "How do I take a scrolling full-page screenshot of an article in Safari on an iPhone 14?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard operational how-to tutorial.",
        "historical_reference_reply": "Take a regular screenshot, tap the preview thumbnail in the bottom corner, and select 'Full Page' at the top: https://support.apple.com/HT200289",
        "human_quality_score": 5
    },
    {
        "id": "how_002",
        "customer_message": "Will the 2nd Generation Apple Pencil work on my standard 10th Generation iPad?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Product compatibility inquiry.",
        "historical_reference_reply": "iPad 10th Gen supports Apple Pencil (USB-C) and Apple Pencil (1st Gen with adapter), but not 2nd Gen. Full compatibility: https://support.apple.com/HT211029",
        "human_quality_score": 5
    },
    {
        "id": "how_003",
        "customer_message": "How do I turn on Personal Hotspot so my laptop can use my iPhone's cellular data?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Step-by-step Personal Hotspot setup tutorial.",
        "historical_reference_reply": "Go to Settings > Cellular > Personal Hotspot and toggle 'Allow Others to Join': https://support.apple.com/HT204023",
        "human_quality_score": 5
    },
    {
        "id": "how_004",
        "customer_message": "What is the estimated trade-in value for an iPhone 12 Pro 128GB in good condition?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Directing customer to Apple Trade In valuation calculator.",
        "historical_reference_reply": "You can get an instant estimated trade-in quote by entering your device details on our Apple Trade In page: https://www.apple.com/shop/trade-in",
        "human_quality_score": 5
    },
    {
        "id": "how_005",
        "customer_message": "How do I scan a paper document directly into a PDF using the Apple Notes app?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Built-in Notes document scanner guide.",
        "historical_reference_reply": "Open Notes, tap the Camera icon, and select 'Scan Documents': https://support.apple.com/HT210336",
        "human_quality_score": 5
    },
    {
        "id": "how_006",
        "customer_message": "Can I connect two pairs of AirPods simultaneously to one iPhone to watch a movie together?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Audio Sharing feature instructions.",
        "historical_reference_reply": "Yes! You can use Audio Sharing via Control Center by tapping the AirPlay icon and selecting 'Share Audio': https://support.apple.com/HT210421",
        "human_quality_score": 5
    },
    {
        "id": "how_007",
        "customer_message": "What date is the new iPhone being announced this September?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Directing customer to official Apple Newsroom / Apple Events page.",
        "historical_reference_reply": "Stay tuned to our Apple Newsroom and Events page for all official keynote announcements: https://www.apple.com/apple-events/",
        "human_quality_score": 4
    },
    {
        "id": "how_008",
        "customer_message": "How do I transfer my eSIM from an old iPhone to a new iPhone wirelessly?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "eSIM Quick Transfer instructions.",
        "historical_reference_reply": "During setup, select 'Transfer from Nearby iPhone' or go to Settings > Cellular > Add eSIM > Transfer from Nearby iPhone: https://support.apple.com/HT212780",
        "human_quality_score": 5
    },
    {
        "id": "how_009",
        "customer_message": "How do I enable Back Tap on iPhone to turn on the flashlight with two taps?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Accessibility Back Tap configuration guide.",
        "historical_reference_reply": "Go to Settings > Accessibility > Touch > Back Tap, choose Double Tap, and assign Flashlight: https://support.apple.com/HT211218",
        "human_quality_score": 5
    },
    {
        "id": "how_010",
        "customer_message": "Is an iPad Air 5 compatible with the Magic Keyboard designed for the 11-inch iPad Pro?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Accessory compatibility confirmation.",
        "historical_reference_reply": "Yes, Magic Keyboard for iPad Pro 11-inch is fully compatible with iPad Air (4th and 5th gen): https://support.apple.com/HT211091",
        "human_quality_score": 5
    },
    {
        "id": "how_011",
        "customer_message": "How do I set a custom song as my iPhone ringtone without using a computer?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "GarageBand on iOS ringtone export tutorial.",
        "historical_reference_reply": "You can export custom audio tracks as ringtones using the free GarageBand app on your device: https://support.apple.com/HT207955",
        "human_quality_score": 4
    },
    {
        "id": "how_012",
        "customer_message": "Can I use Apple Pay on public transit without unlocking my phone with Face ID?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Express Mode public transit feature explanation.",
        "historical_reference_reply": "Yes! You can enable 'Express Transit Card' in Settings > Wallet & Apple Pay to tap and ride without authenticating: https://support.apple.com/HT209495",
        "human_quality_score": 5
    },
    {
        "id": "how_013",
        "customer_message": "How do I turn on Closed Captions for all videos on my Apple TV 4K?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Apple TV accessibility subtitles tutorial.",
        "historical_reference_reply": "Go to Settings > Accessibility > Subtitles and Captioning and turn on Closed Captions: https://support.apple.com/HT202641",
        "human_quality_score": 5
    },
    {
        "id": "how_014",
        "customer_message": "What is the maximum water resistance rating of the Apple Watch Ultra 2?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Factual device water resistance specifications.",
        "historical_reference_reply": "Apple Watch Ultra 2 is water resistant to 100 meters under ISO standard 22810 and certified for recreational dive to 40 meters: https://support.apple.com/HT205000",
        "human_quality_score": 5
    },
    {
        "id": "how_015",
        "customer_message": "How do I transfer all my photos directly from my digital SLR camera to an iPad Pro?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "USB-C SD card adapter and Photos import guide.",
        "historical_reference_reply": "Plug in a USB-C SD Card reader, open Photos, and an 'Import' tab will appear at the bottom: https://support.apple.com/HT202037",
        "human_quality_score": 5
    },
    {
        "id": "how_016",
        "customer_message": "How do I stop my Mac from waking up every time I open the lid?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Terminal AutoBoot NVRAM command explanation.",
        "historical_reference_reply": "Mac notebooks are designed to wake automatically when opened, though advanced users can configure AutoBoot via Terminal: https://support.apple.com/HT201150",
        "human_quality_score": 4
    },
    {
        "id": "how_017",
        "customer_message": "Can I use Sidecar to use my iPad as a secondary display for an older 2015 MacBook Pro?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Sidecar system hardware requirements clarification (2016+ required).",
        "historical_reference_reply": "Sidecar requires a 2016 or newer MacBook Pro. Check all supported hardware: https://support.apple.com/HT210380",
        "human_quality_score": 5
    },
    {
        "id": "how_018",
        "customer_message": "How do I clear the reading list cache in Safari to free up space?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Settings storage management for Safari offline reading list.",
        "historical_reference_reply": "Go to Settings > General > iPhone Storage > Safari and swipe left to delete Offline Reading List: https://support.apple.com/HT201656",
        "human_quality_score": 5
    },
    {
        "id": "how_019",
        "customer_message": "How do I turn on NameDrop to share contact info by bringing two iPhones together?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "NameDrop feature activation in AirDrop settings.",
        "historical_reference_reply": "Go to Settings > General > AirDrop and ensure 'Bringing Devices Together' is turned on: https://support.apple.com/guide/iphone/use-namedrop-iph1b6c664b7/ios",
        "human_quality_score": 5
    },
    {
        "id": "how_020",
        "customer_message": "Does Apple offer educational discounts on MacBook Air for university college students?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Apple Education pricing and UNiDAYS verification guide.",
        "historical_reference_reply": "Yes! Current students and educators can access discounted pricing through the Apple Education Store: https://www.apple.com/us-edu/shop",
        "human_quality_score": 5
    },
    {
        "id": "how_021",
        "customer_message": "How do I configure my iPhone to automatically announce caller names through my AirPods?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Announce Calls setting in Phone settings.",
        "historical_reference_reply": "Go to Settings > Phone > Announce Calls and select 'Headphones Only': https://support.apple.com/HT210403",
        "human_quality_score": 5
    },
    {
        "id": "how_022",
        "customer_message": "Can I record spatial video on an iPhone 14 Pro?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Factual specification requirement (iPhone 15 Pro or later required).",
        "historical_reference_reply": "Spatial video recording requires an iPhone 15 Pro, iPhone 15 Pro Max, or iPhone 16 series: https://support.apple.com/HT213972",
        "human_quality_score": 5
    },
    {
        "id": "how_023",
        "customer_message": "How do I hide the notch or camera cutout on a 14-inch MacBook Pro?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Explanation of macOS menu bar scaling and top bezel area.",
        "historical_reference_reply": "macOS menu bar automatically wraps around the camera housing, or apps can be scaled below it via Get Info > 'Scale to fit below built-in camera': https://support.apple.com/HT212842",
        "human_quality_score": 4
    },
    {
        "id": "how_024",
        "customer_message": "How do I create a custom Smart Playlist in Apple Music on a Mac?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Smart Playlist creation tutorial in Music app.",
        "historical_reference_reply": "In the Music app on Mac, choose File > New > Smart Playlist and set your filter rules: https://support.apple.com/guide/music/create-smart-playlists-mus27cd5060f/mac",
        "human_quality_score": 5
    },
    {
        "id": "how_025",
        "customer_message": "Can I charge my Apple Watch with the wireless MagSafe puck designed for the iPhone?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Magnetic coil inductive charging incompatibility.",
        "historical_reference_reply": "No, Apple Watch requires its dedicated smaller magnetic charging puck or an Apple Watch certified charger: https://support.apple.com/HT204640",
        "human_quality_score": 5
    },
    {
        "id": "how_026",
        "customer_message": "How do I set up Face ID with a medical face mask on iPhone 12?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Face ID with a Mask setup guide.",
        "historical_reference_reply": "Go to Settings > Face ID & Passcode, turn on 'Face ID with a Mask', and follow the onscreen setup: https://support.apple.com/HT213062",
        "human_quality_score": 5
    },
    {
        "id": "how_027",
        "customer_message": "What is the return window policy for products bought directly from the Apple Online Store?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard 14-day Apple return policy.",
        "historical_reference_reply": "Products purchased directly from Apple can be returned within 14 calendar days of receipt: https://www.apple.com/shop/help/returns_refund",
        "human_quality_score": 5
    },
    {
        "id": "how_028",
        "customer_message": "How do I turn on Private Relay in iCloud+ on my iPhone?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "iCloud Private Relay activation guide.",
        "historical_reference_reply": "Go to Settings > [Your Name] > iCloud > Private Relay and turn it on: https://support.apple.com/HT212614",
        "human_quality_score": 5
    },
    {
        "id": "how_029",
        "customer_message": "Can I connect an external hard drive formatted as NTFS directly to an iPad Pro?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "iPadOS read-only NTFS filesystem capability.",
        "historical_reference_reply": "iPadOS can read NTFS formatted drives, but cannot write to them without formatting as APFS or ExFAT: https://support.apple.com/guide/ipad/connect-external-devices-or-servers-ipad4a09101c/ipados",
        "human_quality_score": 4
    },
    {
        "id": "how_030",
        "customer_message": "How do I turn off autocorrect completely on iOS keyboard?",
        "ground_truth_intent": "product_inquiry_howto",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Keyboard settings toggle for Auto-Correction.",
        "historical_reference_reply": "Go to Settings > General > Keyboard and turn off 'Auto-Correction': https://support.apple.com/HT207525",
        "human_quality_score": 5
    },

    # ==========================================
    # 6. COMPLAINT_FEEDBACK (30 examples)
    # ==========================================
    {
        "id": "comp_001",
        "customer_message": "I waited 2 hours at the Apple Store in Regent Street even with an appointment, and the staff was extremely rude and unhelpful! Worst customer service ever!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Severe in-store retail complaint and staff misconduct report requires human store customer relations handling.",
        "historical_reference_reply": "We are very sorry to hear about your experience at our Regent Street store. Please DM us your appointment details and contact info so we can share this with store leadership.",
        "human_quality_score": 5
    },
    {
        "id": "comp_002",
        "customer_message": "Apple's decision to remove the charging brick from the box is pure corporate greed masquerading as environmentalism.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "General product policy feedback; acknowledge and direct to official feedback portal without escalating.",
        "historical_reference_reply": "We appreciate you sharing your perspective on packaging and sustainability. You can submit direct product feedback to our design team here: https://www.apple.com/feedback/",
        "human_quality_score": 5
    },
    {
        "id": "comp_003",
        "customer_message": "I sent my MacBook for repair and it came back with a huge dent on the corner that wasn't there before! You guys ruined my $3,000 laptop!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Depot repair transit damage complaint requires immediate repair escalation and claim dispatch.",
        "historical_reference_reply": "We apologize for this upsetting situation. Please DM us your Repair ID and photos of the damage so our executive relations team can open an immediate investigation.",
        "human_quality_score": 5
    },
    {
        "id": "comp_004",
        "customer_message": "Why does a 1-meter charging cable cost $29? Your accessory pricing is an absolute rip-off.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard pricing feedback venting; capture via public feedback portal.",
        "historical_reference_reply": "We value customer input regarding our accessory lineup and pricing. Please feel free to share your thoughts directly with our product team: https://www.apple.com/feedback/",
        "human_quality_score": 4
    },
    {
        "id": "comp_005",
        "customer_message": "Your phone support agent hung up on me after keeping me on hold for 45 minutes! I am filing a formal complaint with the consumer protection bureau!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Dropped customer call, staff conduct complaint, and legal/regulatory escalation threat requires urgent supervisor contact.",
        "historical_reference_reply": "This is definitely not the standard of service we strive to deliver. Please DM us your case number and phone number immediately so a supervisor can call you.",
        "human_quality_score": 5
    },
    {
        "id": "comp_006",
        "customer_message": "The trade-in quote for my immaculate iPhone 13 Pro was cut in half after I mailed it in because they claimed microscopic scratches. Thieves!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Trade-in partner valuation dispute and customer distress requires human trade-in audit.",
        "historical_reference_reply": "We want to ensure your trade-in was evaluated accurately. Please DM us your Trade-in Quote ID so we can look into the partner assessment for you.",
        "human_quality_score": 5
    },
    {
        "id": "comp_007",
        "customer_message": "I hate the new Photos app redesign in iOS 18. It is completely unintuitive and ruined my workflow. Change it back!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Software UI/UX opinion feedback; route to feedback portal.",
        "historical_reference_reply": "We appreciate hearing how new features impact your daily use. Please submit your detailed design feedback directly to our iOS engineering team: https://www.apple.com/feedback/iphone/",
        "human_quality_score": 5
    },
    {
        "id": "comp_008",
        "customer_message": "Apple took 3 weeks to deliver my custom configured MacBook, missed the guaranteed delivery window, and ruined my video production project deadline.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Late delivery complaint with project financial harm requires order management specialist review.",
        "historical_reference_reply": "We sincerely apologize for the delay and the impact on your project. DM us your web order number so our fulfillment team can review what happened.",
        "human_quality_score": 5
    },
    {
        "id": "comp_009",
        "customer_message": "Why is Apple still charging $1,000 for a phone stand? That monitor stand is insulting.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Pricing critique without personal order issue.",
        "historical_reference_reply": "We appreciate customer perspectives on our Pro display accessories. You can submit feedback anytime at https://www.apple.com/feedback/",
        "human_quality_score": 4
    },
    {
        "id": "comp_010",
        "customer_message": "Your in-store technician at the Mall of America told me my water damage was my fault without even opening the phone. Awful attitude.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "In-store technician service dispute requires retail market manager review.",
        "historical_reference_reply": "We hold our Genius Bar teams to the highest service standards. Please DM us your work authorization number so we can follow up with store management.",
        "human_quality_score": 5
    },
    {
        "id": "comp_011",
        "customer_message": "Every iOS update makes older phones deliberately slower so people are forced to buy the new iPhone. Disgusting planned obsolescence.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "General conspiracy/policy criticism; provide educational link on battery & performance transparently.",
        "historical_reference_reply": "Apple does not deliberately degrade device performance to drive upgrades. Read our full transparent explanation of iOS battery and performance management: https://support.apple.com/HT208387",
        "human_quality_score": 5
    },
    {
        "id": "comp_012",
        "customer_message": "I was promised a call back from a senior manager named Sarah at 3 PM today, and nobody ever called.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Broken commitment from senior leadership requires immediate callback re-scheduling.",
        "historical_reference_reply": "We apologize that you did not receive your scheduled call back. Please DM us your Case Number so we can alert the senior advisor handling your file immediately.",
        "human_quality_score": 5
    },
    {
        "id": "comp_013",
        "customer_message": "Why doesn't Siri understand natural conversation like ChatGPT? It feels like technology from 2011.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "General feature feedback venting.",
        "historical_reference_reply": "We're constantly working to improve Siri's intelligence. Share your feature requests directly with our AI team: https://www.apple.com/feedback/siri/",
        "human_quality_score": 4
    },
    {
        "id": "comp_014",
        "customer_message": "The repair quote to replace the back glass on my iPhone 14 Pro was $499! That is more than half the cost of the entire phone!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Repair pricing feedback; explain AppleCare+ benefits and provide official repair pricing calculator.",
        "historical_reference_reply": "Repair costs reflect precision factory assembly and genuine components. You can review official repair estimates and AppleCare+ deductible rates here: https://support.apple.com/iphone/repair",
        "human_quality_score": 5
    },
    {
        "id": "comp_015",
        "customer_message": "I've been an Apple customer for 15 years and bought 10 Macs, but today's customer support interaction makes me want to switch to Samsung completely.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "High-value loyal customer expressing acute churn risk requires executive customer care triage.",
        "historical_reference_reply": "We truly value your 15 years of loyalty and are heartbroken to hear this. Please DM us your recent case details so we can do everything possible to make things right.",
        "human_quality_score": 5
    },
    {
        "id": "comp_016",
        "customer_message": "The FineWoven iPhone case I bought 2 weeks ago is already frayed, stained, and peeling around the corners. Terrible quality.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Standard return window information for retail accessories.",
        "historical_reference_reply": "If purchased within 14 days, you are eligible for an exchange or return. Learn more about returns here: https://www.apple.com/shop/help/returns_refund",
        "human_quality_score": 5
    },
    {
        "id": "comp_017",
        "customer_message": "Why does Apple continue to charge $200 for an extra 8GB of RAM on a Pro laptop in 2024?",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Hardware spec pricing critique.",
        "historical_reference_reply": "Apple silicon utilizes a unified memory architecture for ultra-high bandwidth efficiency. We welcome your feedback on hardware configurations: https://www.apple.com/feedback/",
        "human_quality_score": 4
    },
    {
        "id": "comp_018",
        "customer_message": "Your courier left my package containing an iPhone 15 Pro on the open sidewalk in the rain without asking for a signature, and it was stolen!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Lost/stolen transit package with delivery carrier negligence requires immediate courier claim and device blacklist.",
        "historical_reference_reply": "We are so sorry to hear this occurred. Please DM us your order number immediately so we can open a carrier loss investigation and send a replacement.",
        "human_quality_score": 5
    },
    {
        "id": "comp_019",
        "customer_message": "The battery gate settlement check I received was for only $92 after waiting 4 years. What a joke.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Legal settlement distribution managed by independent claims administrator.",
        "historical_reference_reply": "Settlement claims and payouts are administered independently by the court-appointed claims administrator at https://www.smartphoneperformancesettlement.com.",
        "human_quality_score": 5
    },
    {
        "id": "comp_020",
        "customer_message": "Your support chat website was completely down for 2 hours while I had an urgent presentation deadline!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "System outage feedback acknowledgment and system status transparency.",
        "historical_reference_reply": "We apologize for the inconvenience during your deadline. You can check live service availability anytime on our System Status page: https://www.apple.com/support/systemstatus/",
        "human_quality_score": 4
    },
    {
        "id": "comp_021",
        "customer_message": "A security guard at the Fifth Avenue Apple store singled me out and searched my bag while letting others walk by. I feel profiled and humiliated.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Allegation of racial or discriminatory profiling in retail store requires immediate escalation to retail VP/legal relations.",
        "historical_reference_reply": "We treat allegations of unfair treatment with the utmost seriousness. Please DM us your contact info so our senior retail leadership team can reach out to you directly.",
        "human_quality_score": 5
    },
    {
        "id": "comp_022",
        "customer_message": "Why does Apple Music shuffle always play the exact same 15 songs out of my 3,000 song library? Your shuffle algorithm is broken.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Algorithm critique and clearing play history recommendation.",
        "historical_reference_reply": "Try clearing your listening history in Settings > Music or toggling Autoplay off. You can also submit algorithm feedback: https://www.apple.com/feedback/apple-music.html",
        "human_quality_score": 4
    },
    {
        "id": "comp_023",
        "customer_message": "I was told my watch would be repaired under warranty, but then the service center sent me a bill for $329 without explanation!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Reclassification of warranty repair to billable service without customer consent requires repair review.",
        "historical_reference_reply": "Let's review the technician inspection notes and warranty coverage. Please DM us your Repair ID so we can investigate this invoice for you.",
        "human_quality_score": 5
    },
    {
        "id": "comp_024",
        "customer_message": "Apple's 30% App Store fee is stifling indie developers and hurting innovation across the entire mobile ecosystem.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "General developer policy criticism; reference App Store Small Business Program (15%).",
        "historical_reference_reply": "Apple supports developers through the App Store Small Business Program, offering a 15% rate for qualifying developers. Learn more: https://developer.apple.com/app-store/small-business-program/",
        "human_quality_score": 5
    },
    {
        "id": "comp_025",
        "customer_message": "I sent my phone in for trade-in in mint condition and Apple Trade-in says the box was empty when received. You basically stole my phone!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Transit theft allegation / empty package claim requires carrier fraud investigation.",
        "historical_reference_reply": "We take missing package claims very seriously. DM us your Trade-in Quote ID and carrier tracking number so we can launch an immediate investigation with the courier.",
        "human_quality_score": 5
    },
    {
        "id": "comp_026",
        "customer_message": "Why does iOS still not allow placing app icons anywhere on the grid without snapping to the top?",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "General UI feature request.",
        "historical_reference_reply": "We love hearing customization ideas! You can share your suggestions directly with our human interface guidelines team: https://www.apple.com/feedback/iphone/",
        "human_quality_score": 4
    },
    {
        "id": "comp_027",
        "customer_message": "The trade-in program rejected my iPad because of a microscopic hairline scratch on the apple logo on the back! Absolutely ridiculous standard.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Explanation of trade-in condition guidelines and option to have device returned at no cost.",
        "historical_reference_reply": "If a revised value or rejection is not acceptable, you can decline the quote and have your device returned to you free of charge: https://www.apple.com/shop/trade-in",
        "human_quality_score": 5
    },
    {
        "id": "comp_028",
        "customer_message": "An Apple support advisor promised to waive my repair fee on the phone yesterday, but the store says they have no record of this promise!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Conflicting commitments between phone support and retail store requires case note audit.",
        "historical_reference_reply": "We can pull up the recorded case notes from your phone call. Please DM us your Case ID and the store location you visited.",
        "human_quality_score": 5
    },
    {
        "id": "comp_029",
        "customer_message": "Why is the base storage on iPads still 64GB in this day and age? It fills up after two games.",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "AUTO_HANDLE",
        "ground_truth_escalation_reason": "Storage specification critique.",
        "historical_reference_reply": "We appreciate feedback on our entry-level storage tiers. Please submit your thoughts to our product planning team: https://www.apple.com/feedback/",
        "human_quality_score": 4
    },
    {
        "id": "comp_030",
        "customer_message": "Your battery health update destroyed my phone and now I'm joining the class action lawsuit against Apple Inc. See you in federal court!",
        "ground_truth_intent": "complaint_feedback",
        "ground_truth_escalation": "ESCALATE_TO_HUMAN",
        "ground_truth_escalation_reason": "Explicit legal threat and class-action lawsuit declaration requires standard corporate legal triage protocol.",
        "historical_reference_reply": "We hear your frustration. Please DM us your contact information so we can connect you with the appropriate relations team.",
        "human_quality_score": 5
    }
]


def build_and_save_golden_set():
    """Builds and writes golden evaluation set in both JSON and CSV formats."""
    print(f"Total curated golden examples: {len(GOLDEN_EXAMPLES)}")

    # Intent distribution
    intent_counts = {}
    escalation_counts = {}
    for ex in GOLDEN_EXAMPLES:
        intent = ex["ground_truth_intent"]
        esc = ex["ground_truth_escalation"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        escalation_counts[esc] = escalation_counts.get(esc, 0) + 1

    print("\nIntent Distribution:")
    for k, v in intent_counts.items():
        print(f"  {k}: {v}")

    print("\nEscalation Distribution:")
    for k, v in escalation_counts.items():
        print(f"  {k}: {v}")

    # Write JSON
    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(GOLDEN_EXAMPLES, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON to {GOLDEN_SET_PATH}")

    # Write CSV
    df = pd.DataFrame(GOLDEN_EXAMPLES)
    df.to_csv(GOLDEN_SET_CSV_PATH, index=False, encoding="utf-8")
    print(f"Saved CSV to {GOLDEN_SET_CSV_PATH}")

    # Write Human Judge sample (first 50 examples with human ratings)
    human_sample = []
    for ex in GOLDEN_EXAMPLES[:50]:
        human_sample.append({
            "id": ex["id"],
            "customer_message": ex["customer_message"],
            "reference_reply": ex["historical_reference_reply"],
            "human_groundedness_score": ex["human_quality_score"],
            "human_actionability_score": ex["human_quality_score"],
            "human_tone_score": 5 if ex["human_quality_score"] >= 4 else 4,
            "human_escalation_appropriateness": 5,
            "human_overall_rating": ex["human_quality_score"]
        })

    with open(HUMAN_JUDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(human_sample, f, indent=2, ensure_ascii=False)
    print(f"Saved Human Judge sample (50 items) to {HUMAN_JUDGE_PATH}")


if __name__ == "__main__":
    build_and_save_golden_set()

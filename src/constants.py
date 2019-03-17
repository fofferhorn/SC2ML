# Mapping of unit_id to index in list.
protoss_unit_mapper = {
    # Units
    84: 0,      # Probe
    73: 1,      # Zealot
    77: 2,      # Sentry
    311: 3,     # Adept
    74: 4,      # Stalker
    75: 5,      # HighTemplar
    76: 6,      # DarkTemplar
    141: 7,     # Archon
    4: 8,       # Colossus
    694: 9,     # Disruptor
    82: 10,     # Observer
    83: 11,     # Immortal
    81: 12,     # WarpPrism
    78: 13,     # Phoenix
    495: 14,    # Oracle
    80: 15,     # VoidRay
    496: 16,    # Tempest
    79: 17,     # Carrier
    85: 18,     # Interceptor
    10: 19,     # Mothership
    488: 20,    # MothershipCore

    # Buildings
    59: 21,     # Nexus
    60: 22,     # Pylon
    61: 23,     # Assimilator
    62: 24,     # Gateway
    133: 25,    # WarpGate
    72: 26,     # CyberneticsCore
    65: 27,     # TwilightCouncil
    68: 28,     # TemplarArchive
    69: 29,     # DarkShrine
    63: 30,     # Forge
    66: 31,     # PhotonCannon
    1910: 32,   # ShieldBattery
    70: 33,     # RoboticsBay
    71: 34,     # RoboticsFacility
    67: 35,     # Stargate
    64: 36,     # FleetBeacon

    # Abilities
    801: 37,    # AdeptPhaseShift
    135: 38,    # ForceField
    1911: 39,   # ObserverSurveillanceMode
    733: 40,    # DisruptorPhased
    136: 41,    # WarpPrismPhasing
    894: 42,    # PylonOvercharged
    732: 43     # StasisTrap
}

# Mapping of upgrades to indexes in list
protoss_upgrade_mapper = {
    1: 0,       # CARRIERLAUNCHSPEEDUPGRADE
    39: 1,      # PROTOSSGROUNDWEAPONSLEVEL1
    40: 2,      # PROTOSSGROUNDWEAPONSLEVEL2
    41: 3,      # PROTOSSGROUNDWEAPONSLEVEL3
    42: 4,      # PROTOSSGROUNDARMORSLEVEL1
    43: 5,      # PROTOSSGROUNDARMORSLEVEL2
    44: 6,      # PROTOSSGROUNDARMORSLEVEL3
    45: 7,      # PROTOSSSHIELDSLEVEL1
    46: 8,      # PROTOSSSHIELDSLEVEL2
    47: 9,      # PROTOSSSHIELDSLEVEL3
    48: 10,     # OBSERVERGRAVITICBOOSTER
    49: 11,     # GRAVITICDRIVE
    50: 12,     # EXTENDEDTHERMALLANCE
    52: 13,     # PSISTORMTECH
    78: 14,     # PROTOSSAIRWEAPONSLEVEL1
    79: 15,     # PROTOSSAIRWEAPONSLEVEL2
    80: 16,     # PROTOSSAIRWEAPONSLEVEL3
    81: 17,     # PROTOSSAIRARMORSLEVEL1
    82: 18,     # PROTOSSAIRARMORSLEVEL2
    83: 19,     # PROTOSSAIRARMORSLEVEL3
    84: 20,     # WARPGATERESEARCH
    86: 21,     # CHARGE
    87: 22,     # BLINKTECH
    99: 23,     # PHOENIXRANGEUPGRADE
    130: 24,    # ADEPTPIERCINGATTACK
    141: 25     # DARKTEMPLARBLINKUPGRADE
}

# # Mapping of macro actions to the buildings/units/upgrades they will turn into. 
# protoss_action_to_unit_mapper = {
#     # Build
#     882: 23,    # BUILD_ASSIMILATOR
#     894: 26,    # BUILD_CYBERNETICSCORE
#     891: 29,    # BUILD_DARKSHRINE
#     885: 36,    # BUILD_FLEETBEACON
#     884: 30,    # BUILD_FORGE
#     883: 24,    # BUILD_GATEWAY
#     1042: 18,   # BUILD_INTERCEPTORS
#     880: 21,    # BUILD_NEXUS
#     887: 31,    # BUILD_PHOTONCANNON
#     881: 22,    # BUILD_PYLON
#     892: 33,   # BUILD_ROBOTICSBAY
#     893: 34,   # BUILD_ROBOTICSFACILITY
#     895: 32,   # BUILD_SHIELDBATTERY
#     889: 35,   # BUILD_STARGATE
#     890: 28,   # BUILD_TEMPLARARCHIVE
#     886: 27,   # BUILD_TWILIGHTCOUNCIL

#     # Morph
#     1766: 7,  # MORPH_ARCHON
#     1520: 24,  # MORPH_GATEWAY
#     1847: 19,  # MORPH_MOTHERSHIP
#     1518: 25,  # MORPH_WARPGATE

#     # Research
#     1594: 24,  # RESEARCH_ADEPTRESONATINGGLAIVES
#     1593: 22,  # RESEARCH_BLINK
#     1592: 21,  # RESEARCH_CHARGE
#     1097: 12,  # RESEARCH_EXTENDEDTHERMALLANCE
#     1093: 10,  # RESEARCH_GRAVITICBOOSTER
#     1094: 11,  # RESEARCH_GRAVITICDRIVE
#     44: 0,    # RESEARCH_INTERCEPTORGRAVITONCATAPULT
#     46: 23,    # RESEARCH_PHOENIXANIONPULSECRYSTALS
#     1565: 17,  # RESEARCH_PROTOSSAIRARMORLEVEL1
#     1566: 18,  # RESEARCH_PROTOSSAIRARMORLEVEL2
#     1567: 19,  # RESEARCH_PROTOSSAIRARMORLEVEL3
#     1562: 14,  # RESEARCH_PROTOSSAIRWEAPONSLEVEL1
#     1563: 15,  # RESEARCH_PROTOSSAIRWEAPONSLEVEL2
#     1564: 16,  # RESEARCH_PROTOSSAIRWEAPONSLEVEL3
#     1065: 4,  # RESEARCH_PROTOSSGROUNDARMORLEVEL1
#     1066: 5,  # RESEARCH_PROTOSSGROUNDARMORLEVEL2
#     1067: 6,  # RESEARCH_PROTOSSGROUNDARMORLEVEL3
#     1062: 1,  # RESEARCH_PROTOSSGROUNDWEAPONSLEVEL1
#     1063: 2,  # RESEARCH_PROTOSSGROUNDWEAPONSLEVEL2
#     1064: 3,  # RESEARCH_PROTOSSGROUNDWEAPONSLEVEL3
#     1068: 7,  # RESEARCH_PROTOSSSHIELDSLEVEL1
#     1069: 8,  # RESEARCH_PROTOSSSHIELDSLEVEL2
#     1070: 9,  # RESEARCH_PROTOSSSHIELDSLEVEL3
#     1126: 13,  # RESEARCH_PSISTORM
#     2720: 25,  # RESEARCH_SHADOWSTRIKE
#     1568: 20,  # RESEARCH_WARPGATE

#     # Train
#     922: 3,   # TRAIN_ADEPT
#     948: 17,   # TRAIN_CARRIER
#     978: 8,   # TRAIN_COLOSSUS
#     920: 6,   # TRAIN_DARKTEMPLAR
#     994: 9,   # TRAIN_DISRUPTOR
#     919: 5,   # TRAIN_HIGHTEMPLAR
#     979: 11,   # TRAIN_IMMORTAL
#     110: 19,   # TRAIN_MOTHERSHIP
#     1853: 20,  # TRAIN_MOTHERSHIPCORE
#     977: 10,   # TRAIN_OBSERVER
#     954: 14,   # TRAIN_ORACLE
#     946: 13,   # TRAIN_PHOENIX
#     1006: 0,  # TRAIN_PROBE
#     921: 2,   # TRAIN_SENTRY
#     917: 4,   # TRAIN_STALKER
#     955: 16,   # TRAIN_TEMPEST
#     950: 15,   # TRAIN_VOIDRAY
#     976: 12,   # TRAIN_WARPPRISM
#     916: 1,   # TRAIN_ZEALOT

#     # TrainWarp
#     1419: 3,  # TRAINWARP_ADEPT
#     1417: 6,  # TRAINWARP_DARKTEMPLAR
#     1416: 5,  # TRAINWARP_HIGHTEMPLAR
#     1418: 2,  # TRAINWARP_SENTRY
#     1414: 4,  # TRAINWARP_STALKER
#     1413: 1,  # TRAINWARP_ZEALOT

#     # # Cancel
#     # 3659,  # CANCEL
#     # 313,   # CANCELSLOT_ADDON
#     # 305,   # CANCELSLOT_QUEUE1
#     # 307,   # CANCELSLOT_QUEUE5
#     # 309,   # CANCELSLOT_QUEUECANCELTOSELECTION
#     # 1832,  # CANCELSLOT_QUEUEPASSIVE
#     # 314,   # CANCEL_BUILDINPROGRESS
#     # 3671,  # CANCEL_LAST
#     # 1848,  # CANCEL_MORPHMOTHERSHIP
#     # 304,   # CANCEL_QUEUE1
#     # 306,   # CANCEL_QUEUE5
#     # 312,   # CANCEL_QUEUEADDON
#     # 308,   # CANCEL_QUEUECANCELTOSELECTION
#     # 1831,  # CANCEL_QUEUEPASIVE
#     # 1833,  # CANCEL_QUEUEPASSIVECANCELTOSELECTION

#     # # Stop
#     # 3665,  # STOP
#     # 2057,  # STOP_BUILDING
#     # 4,     # STOP_STOP
# }

# Mapping of macro actions to the buildings/units/upgrades they will turn into. 
protoss_action_to_unit_mapper = {
    # Build
    882: 0,    # BUILD_ASSIMILATOR
    894: 1,    # BUILD_CYBERNETICSCORE
    891: 2,    # BUILD_DARKSHRINE
    885: 3,    # BUILD_FLEETBEACON
    884: 4,    # BUILD_FORGE
    883: 5,    # BUILD_GATEWAY
    1042: 6,   # BUILD_INTERCEPTORS
    880: 7,    # BUILD_NEXUS
    887: 8,    # BUILD_PHOTONCANNON
    881: 9,    # BUILD_PYLON
    892: 10,   # BUILD_ROBOTICSBAY
    893: 11,   # BUILD_ROBOTICSFACILITY
    895: 12,   # BUILD_SHIELDBATTERY
    889: 13,   # BUILD_STARGATE
    890: 14,   # BUILD_TEMPLARARCHIVE
    886: 15,   # BUILD_TWILIGHTCOUNCIL

    # Morph
    1766: 16,  # MORPH_ARCHON
    1520: 5,  # MORPH_GATEWAY
    1847: 17,  # MORPH_MOTHERSHIP
    1518: 18,  # MORPH_WARPGATE

    # Research
    1594: 19,  # RESEARCH_ADEPTRESONATINGGLAIVES
    1593: 20,  # RESEARCH_BLINK
    1592: 21,  # RESEARCH_CHARGE
    1097: 22,  # RESEARCH_EXTENDEDTHERMALLANCE
    1093: 23,  # RESEARCH_GRAVITICBOOSTER
    1094: 24,  # RESEARCH_GRAVITICDRIVE
    44: 25,    # RESEARCH_INTERCEPTORGRAVITONCATAPULT
    46: 26,    # RESEARCH_PHOENIXANIONPULSECRYSTALS
    1565: 27,  # RESEARCH_PROTOSSAIRARMORLEVEL1
    1566: 27,  # RESEARCH_PROTOSSAIRARMORLEVEL2
    1567: 27,  # RESEARCH_PROTOSSAIRARMORLEVEL3
    1562: 28,  # RESEARCH_PROTOSSAIRWEAPONSLEVEL1
    1563: 28,  # RESEARCH_PROTOSSAIRWEAPONSLEVEL2
    1564: 28,  # RESEARCH_PROTOSSAIRWEAPONSLEVEL3
    1065: 29,  # RESEARCH_PROTOSSGROUNDARMORLEVEL1
    1066: 29,  # RESEARCH_PROTOSSGROUNDARMORLEVEL2
    1067: 29,  # RESEARCH_PROTOSSGROUNDARMORLEVEL3
    1062: 30,  # RESEARCH_PROTOSSGROUNDWEAPONSLEVEL1
    1063: 30,  # RESEARCH_PROTOSSGROUNDWEAPONSLEVEL2
    1064: 30,  # RESEARCH_PROTOSSGROUNDWEAPONSLEVEL3
    1068: 31,  # RESEARCH_PROTOSSSHIELDSLEVEL1
    1069: 31,  # RESEARCH_PROTOSSSHIELDSLEVEL2
    1070: 31,  # RESEARCH_PROTOSSSHIELDSLEVEL3
    1126: 32,  # RESEARCH_PSISTORM
    2720: 33,  # RESEARCH_SHADOWSTRIKE
    1568: 34,  # RESEARCH_WARPGATE

    # Train
    922: 35,   # TRAIN_ADEPT
    948: 36,   # TRAIN_CARRIER
    978: 37,   # TRAIN_COLOSSUS
    920: 38,   # TRAIN_DARKTEMPLAR
    994: 39,   # TRAIN_DISRUPTOR
    919: 40,   # TRAIN_HIGHTEMPLAR
    979: 41,   # TRAIN_IMMORTAL
    110: 42,   # TRAIN_MOTHERSHIP
    1853: 43,  # TRAIN_MOTHERSHIPCORE
    977: 44,   # TRAIN_OBSERVER
    954: 45,   # TRAIN_ORACLE
    946: 46,   # TRAIN_PHOENIX
    1006: 47,  # TRAIN_PROBE
    921: 48,   # TRAIN_SENTRY
    917: 49,   # TRAIN_STALKER
    955: 50,   # TRAIN_TEMPEST
    950: 51,   # TRAIN_VOIDRAY
    976: 52,   # TRAIN_WARPPRISM
    916: 53,   # TRAIN_ZEALOT

    # TrainWarp
    1419: 35,  # TRAINWARP_ADEPT
    1417: 38,  # TRAINWARP_DARKTEMPLAR
    1416: 40,  # TRAINWARP_HIGHTEMPLAR
    1418: 48,  # TRAINWARP_SENTRY
    1414: 49,  # TRAINWARP_STALKER
    1413: 53,  # TRAINWARP_ZEALOT
}

output_to_action = [
    # Build
    0,  # ASSIMILATOR
    1,  # CYBERNETICSCORE
    2,  # DARKSHRINE
    3,  # FLEETBEACON
    4,  # FORGE
    5,  # GATEWAY
    6,  # INTERCEPTORS
    7,  # NEXUS
    8,  # PHOTONCANNON
    9,  # PYLON
    10, # ROBOTICSBAY
    11, # ROBOTICSFACILITY
    12, # SHIELDBATTERY
    13, # STARGATE
    14, # TEMPLARARCHIVE
    15, # TWILIGHTCOUNCIL

    # Morph
    16, # ARCHON
    17, # MOTHERSHIP
    18, # WARPGATE

    # Research
    19, # ADEPTRESONATINGGLAIVES
    20, # BLINK
    21, # CHARGE
    22, # EXTENDEDTHERMALLANCE
    23, # GRAVITICBOOSTER
    24, # GRAVITICDRIVE
    25, # INTERCEPTORGRAVITONCATAPULT
    26, # PHOENIXANIONPULSECRYSTALS
    27, # PROTOSSAIRARMOR
    28, # PROTOSSAIRWEAPONS
    29, # PROTOSSGROUNDARMOR
    30, # PROTOSSGROUNDWEAPONS
    31, # PROTOSSSHIELDS
    32, # PSISTORM
    33, # SHADOWSTRIKE
    34, # WARPGATE

    # Train
    35, # ADEPT
    36, # CARRIER
    37, # COLOSSUS
    38, # DARKTEMPLAR
    39, # DISRUPTOR
    40, # HIGHTEMPLAR
    41, # IMMORTAL
    42, # MOTHERSHIP
    43, # MOTHERSHIPCORE
    44, # OBSERVER
    45, # ORACLE
    46, # PHOENIX
    47, # PROBE
    48, # SENTRY
    49, # STALKER
    50, # TEMPEST
    51, # VOIDRAY
    52, # WARPPRISM
    53  # ZEALOT
]

macro_actions = [
    "Build", 
    "Cancel", 
    "Morph", 
    "Research", 
    "Stop", 
    "Train", 
    "TrainWarp"
]

# The amount of different macro actions a player can take. 
protoss_macro_actions = [
    # Buildings
    'BUILD_ASSIMILATOR',
    'BUILD_CYBERNETICSCORE',
    'BUILD_DARKSHRINE',
    'BUILD_FLEETBEACON',
    'BUILD_FORGE',
    'BUILD_GATEWAY',
    'BUILD_INTERCEPTORS',
    'BUILD_NEXUS',
    'BUILD_PHOTONCANNON',
    'BUILD_PYLON',
    'BUILD_ROBOTICSBAY',
    'BUILD_ROBOTICSFACILITY',
    'BUILD_SHIELDBATTERY',
    'BUILD_STARGATE',
    'BUILD_STASISTRAP',
    'BUILD_TEMPLARARCHIVE',
    'BUILD_TWILIGHTCOUNCIL',

    # Morph
    'MORPH_ARCHON',
    'MORPH_GATEWAY',
    'MORPH_MOTHERSHIP',
    'MORPH_WARPGATE',

    # Research
    'RESEARCH_ADEPTRESONATINGGLAIVES',
    'RESEARCH_BLINK',
    'RESEARCH_CHARGE',
    'RESEARCH_EXTENDEDTHERMALLANCE',
    'RESEARCH_GRAVITICBOOSTER',
    'RESEARCH_GRAVITICDRIVE',
    'RESEARCH_INTERCEPTORGRAVITONCATAPULT',
    'RESEARCH_PHOENIXANIONPULSECRYSTALS',
    'RESEARCH_PROTOSSAIRARMOR',
    'RESEARCH_PROTOSSAIRARMORLEVEL1',
    'RESEARCH_PROTOSSAIRARMORLEVEL2',
    'RESEARCH_PROTOSSAIRARMORLEVEL3',
    'RESEARCH_PROTOSSAIRWEAPONS',
    'RESEARCH_PROTOSSAIRWEAPONSLEVEL1',
    'RESEARCH_PROTOSSAIRWEAPONSLEVEL2',
    'RESEARCH_PROTOSSAIRWEAPONSLEVEL3',
    'RESEARCH_PROTOSSGROUNDARMOR',
    'RESEARCH_PROTOSSGROUNDARMORLEVEL1',
    'RESEARCH_PROTOSSGROUNDARMORLEVEL2',
    'RESEARCH_PROTOSSGROUNDARMORLEVEL3',
    'RESEARCH_PROTOSSGROUNDWEAPONS',
    'RESEARCH_PROTOSSGROUNDWEAPONSLEVEL1',
    'RESEARCH_PROTOSSGROUNDWEAPONSLEVEL2',
    'RESEARCH_PROTOSSGROUNDWEAPONSLEVEL3',
    'RESEARCH_PROTOSSSHIELDS',
    'RESEARCH_PROTOSSSHIELDSLEVEL1',
    'RESEARCH_PROTOSSSHIELDSLEVEL2',
    'RESEARCH_PROTOSSSHIELDSLEVEL3',
    'RESEARCH_PSISTORM',
    'RESEARCH_RAPIDFIRELAUNCHERS',
    'RESEARCH_SHADOWSTRIKE',
    'RESEARCH_WARPGATE',
    
    # Train
    'TRAIN_ADEPT',
    'TRAIN_CARRIER',
    'TRAIN_COLOSSUS',
    'TRAIN_DARKTEMPLAR',
    'TRAIN_DISRUPTOR',
    'TRAIN_HIGHTEMPLAR',
    'TRAIN_IMMORTAL',
    'TRAIN_MOTHERSHIP',
    'TRAIN_MOTHERSHIPCORE',
    'TRAIN_OBSERVER',
    'TRAIN_ORACLE',
    'TRAIN_PHOENIX',
    'TRAIN_PROBE',
    'TRAIN_SENTRY',
    'TRAIN_STALKER',
    'TRAIN_TEMPEST',
    'TRAIN_VOIDRAY',
    'TRAIN_WARPPRISM',
    'TRAIN_ZEALOT'
]
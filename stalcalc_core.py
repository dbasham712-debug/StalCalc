# stalcalc_core.py
import pandas as pd
import numpy as np
import time
from typing import Optional
# ===================== DATA =====================
red_items = [
    {
        "name": "Red Gum",
        "Quality": 160,
        "Health Regeneration": 3.95,
        "Radiation": 6.32,
        "Temperature": 2.66,
        "Reaction to chemical burns": 9.78
    },
    {
        "name": "Red Dumbbell",
        "Quality": 160,
        "Vitality": 1.25,
        "Healing Effectiveness": 32.03,
        "Radiation": -2.5
    },
    {
        "name": "Red Hoop",
        "Quality": 160,
        "Health Regeneration": 8.32,
        "Bullet Resistance": 25.38,
        "Psy-Emissions": -2.5
    },
    {
        "name": "Red Dark Crystal",
        "Quality": 160,
        "Health Regeneration": 6.45,
        "Stamina Regeneration": 6.45,
        "Radiation": 2.66,
        "Psy-Emissions": -0.63,
        "Vitality": 2.08
    },
    {
        "name": "Red Timber",
        "Quality": 160,
        "Bullet Resistance": 22.46,
        "Radiation": 2.66,
        "Temperature": -0.63,
        "Health Regeneration": 3.95
    },
    {
        "name": "Red Black Hole",
        "Quality": 160,
        "Bullet Resistance": 28.08,
        "Vitality": 1.46,
        "Health Regeneration": 4.78,
        "Psy-Emissions": -2.5
    },
    {
        "name": "Red Larva",
        "Quality": 160,
        "Healing Effectiveness": 19.97,
        "Biological Infection": -0.63,
        "Temperature": 2.66,
        "Vitality": 2.5
    },
    {
        "name": "Red Mug",
        "Quality": 160,
        "Health Regeneration": 3.95,
        "Vitality": 3.33,
        "Radiation": -0.63,
        "Biological Infection": 2.66
    },
    {
        "name": "Red Snares",
        "Quality": 160,
        "Bullet Resistance": 14.14,
        "Psy-Emissions": 8.49
    },
    {
        "name": "Red Opal",
        "Quality": 160,
        "Vitality": -2.9,
        "Bullet Resistance": 53.66,
        "Temperature": 0.8,
        "Frost": -1
    },
    #{
    #    "name": "Red Leaded Glass",
    #    "Quality": 160,
    #    "Radiation": 2.66,
    #    "Temperature": 0.40,
    #    "Psy-Emissions": 2.66,
    #    "Frost": -0.50,
    #    "Burning": -0.62,
    #    "Reaction to burns": 4.58,
    #    "Vitality": 1.25
    #},
    #{
    #    "name": "Red Onion",
    #    "Quality": 160,
    #    "Stamina": 21.63,
    #    "Bullet Resistance": 19.55,
    #    "Radiation": -0.63,
    #    "Psy-Emissions": 2.66,
    #    "Bleeding Protection": 9.57
    #},
    {
        "name": "Red Stress Fest",
        "Quality": 160,
        "Movement Speed": 3.95,
        "Healing Effectiveness": 28.29,
        "Temperature": -1.25,
        "Carry Weight": 18.10
    },
    #{
    #    "name": "Red Phlegm",
    #    "Quality": 160,
    #    "Healing Effectiveness": 23.09,
    #    "Bleeding Protection": 15.81,
    #    "Biological Infection": -1.25,
    #    "Bleeding": -1.66
    #},
    #{
    #    "name": "Red Lard",
    #    "Quality": 160,
    #    "Health Regeneration": 6.24,
    #    "Healing Effectiveness": 16.22,
    #    "Radiation": -1.25,
    #    "Reaction to chemical burns": 5.2
    #},
    #{
    #    "name": "Red Heart",
    #    "Quality": 160,
    #    "Health Regeneration": 7.07,
    #    "Stamina Regeneration": 7.07,
    #    "Radiation": -1.25,
    #    "Healing Effectiveness": 14.14
    #},
    #{
    #    "name": "Red Rose",
    #    "Quality": 160,
    #    "Bullet Resistance": 24.13,
    #    "Explosion Protection": 18.72,
    #    "Psy-Emissions": -1.25,
    #    "Carry Weight": 13.52
    #},
    {
        "name": "Red Radiator",
        "Quality": 160,
        "Movement Speed": 2.91,
        "Temperature": 6.32,
        "Psy-Emissions": 2.66
    },
    {
        "name": "Red Gills",
        "Quality": 160,
        "Vitality": 3.74,
        "Health Regeneration": 8.32,
        "Radiation": -2.5,
        "Stamina Regeneration": 6.66
    },
    {
        "name": "Red Scrubber",
        "Quality": 160,
        "Healing Effectiveness": 14.98,
        "Radiation": 2.66,
        "Biological Infection": 6.32,
        "Reaction to laceration": 8.11
    },
    {
        "name": "Red Fossil",
        "Quality": 160,
        "Vitality": 5.82,
        "Stamina Regeneration": 7.49,
        "Biological Infection": -2.50,
        "Periodic Healing": 1.46
    },
    {
        "name": "Red Shard",
        "Quality": 160,
        "Vitality": 5.41,
        "Healing Effectiveness": 22.46,
        "Biological Infection": -2.5,
        "Stamina Regeneration": 6.66
    },
    {
        "name": "Red Atom",
        "Quality": 160,
        "Stamina Regeneration": 9.57,
        "Radiation": 2.66,
        "Temperature": 2.66,
        "Biological Infection": 2.66,
        "Psy-Emissions": 2.66,
        "Bleeding": -2.50,
        "Vitality": 2.08
    },
    {
        "name": "Red Cursed Rose",
        "Quality": 160,
        "Bullet Resistance": 32.24,
        "Explosion Protection": 24.75,
        "Radiation": -1.25,
        "Stamina Regeneration": 6.03
    },
    {
        "name": "Red Proto-Onion",
        "Quality": 160,
        "Stamina Regeneration": 7.49,
        "Healing Effectiveness": 13.31,
        "Radiation": 6.32,
        "Health Regeneration": 2.7
    },
    {
        "name": "Red Veiner",
        "Quality": 160,
        "Carry Weight": 17.06,
        "Bullet Resistance": 22.46,
        "Radiation": -0.63,
        "Psy-Emissions": 2.66,
        "Stamina Regeneration": 4.16
    },
    {
        "name": "Red Prism",
        "Quality": 160,
        "Bullet Resistance": 40.35,
        "Explosion Protection": 31.20,
        "Psy-Emissions": -2.5,
        "Stamina Regeneration": 7.49
    },
    {
        "name": "Red Scallop",
        "Quality": 175,
        "Bullet Resistance": 28.08,
        "Explosion Protection": 21.63,
        "Radiation": -1.25,
        "Bleeding": -1.87
    },
    {
        "name": "Red Frame",
        "Quality": 160,
        "Carry Weight": 25.79,
        "Stamina Regeneration": 6.24,
        "Radiation": 2.66,
        "Temperature": 2.66,
        "Psy-Emissions": 2.66,
        "Biological Infection": 2.66
    },
    {
        "name": "Red Rime",
        "Quality": 160,
        "Bullet Resistance": 28.08,
        "Laceration Protection": 29.74,
        "Temperature": 0.40,
        "Frost": -0.5,
        "Explosion Protection": 16.22
    },
    {
        "name": "Red Chilly",
        "Quality": 160,
        "Vitality": 7.49,
        "Temperature": 0.8,
        "Frost": -1,
        "Burning": -1.25,
        "Explosion Protection": 30.99
    },
    {
        "name": "Red Heel",
        "Quality": 160,
        "Healing Effectiveness": 42.43,
        "Temperature": 0.80,
        "Frost": -1,
        "Reaction to laceration": 11.23,
        "Reaction to burns": 6.86
    },
    {
        "name": "Red Firebird",
        "Quality": 160,
        "Health Regeneration": 5.41,
        "Healing Effectiveness": 28.29,
        "Temperature": -1.25,
        "Bleeding": -2.08
    },
    {
        "name": "Red Viburnum Branch",
        "Quality": 160,
        "Vitality": 4.16,
        "Healing Effectiveness": 35.57,
        "Radiation": -2.5,
        "Stamina": 33.7
    },
    {
        "name": "Red Transformer",
        "Quality": 160,
        "Stamina": 38.69,
        "Carry Weight": 18.10,
        "Biological Infection": 2.66,
        "Psy-Emissions": 6.32
    },
    {
        "name": "Red Tallow",
        "Quality": 160,
        "Health Regeneration": 5.2,
        "Healing Effectiveness": 28.29,
        "Radiation": -1.25,
        "Temperature": 2.66
    }
    #{
    #    "name": "Green Bubblegum",
    #    "Quality": 115,
    #    "Bullet Resistance": 38.57,
    #    "Explosion Protection": 29.75,
    #    "Temperature": -2.50
    #}
]
pink_items = [
    {
        "name": "Pink Gum",
        "Quality": 145,
        "Health Regeneration": 3.77,
        "Radiation": 5.73,
        "Temperature": 2.41,
        "Reaction to chemical burns": 8.86
    },
    {
        "name": "Pink Dark Crystal",
        "Quality": 145,
        "Health Regeneration": 5.84,
        "Stamina Regeneration": 5.84,
        "Radiation": 2.41,
        "Psy-Emissions": -0.63,
        "Vitality": 1.89
    },
    {
        "name": "Pink Radiator",
        "Quality": 145,
        "Movement Speed": 2.64,
        "Temperature": 5.73,
        "Psy-Emissions": 2.41,
        "Reaction to burns": 6.03
    },
    {
        "name": "Pink Gills",
        "Quality": 145,
        "Vitality": 3.39,
        "Health Regeneration": 8.11,
        "Radiation": -2.5,
        "Stamina Regeneration": 6.03
    },
    {
        "name": "Pink Scrubber",
        "Quality": 145,
        "Healing Effectiveness": 14.7,
        "Radiation": 2.41,
        "Biological Infection": 5.73,
        "Reaction to laceration": 7.35
    },
    {
        "name": "Pink Fossil",
        "Quality": 145,
        "Vitality": 5.28,
        "Stamina Regeneration": 9.05,
        "Biological Infection": -2.50,
        "Periodic Healing": 1.13
    },
    {
        "name": "Pink Shard",
        "Quality": 145,
        "Vitality": 4.90,
        "Healing Effectiveness": 21.87,
        "Biological Infection": -2.5,
        "Stamina Regeneration": 6.03
    },
    {
        "name": "Pink Atom",
        "Quality": 145,
        "Stamina Regeneration": 8.67,
        "Radiation": 2.41,
        "Temperature": 2.41,
        "Biological Infection": 2.41,
        "Psy-Emissions": 2.41,
        "Bleeding": -2.26,
        "Vitality": 1.89
    },
    {
        "name": "Pink Cursed Rose",
        "Quality": 145,
        "Bullet Resistance": 29.22,
        "Explosion Protection": 22.43,
        "Radiation": -1.25,
        "Stamina Regeneration": 5.47
    },
    {
        "name": "Pink Proto-Onion",
        "Quality": 145,
        "Stamina Regeneration": 6.79,
        "Healing Effectiveness": 13.20,
        "Radiation": 5.73,
        "Health Regeneration": 2.64
    },
    {
        "name": "Pink Veiner",
        "Quality": 145,
        "Carry Weight": 15.46,
        "Bullet Resistance": 20.36,
        "Radiation": -0.63,
        "Psy-Emissions": 2.41,
        "Stamina Regeneration": 3.77
    },
    {
        "name": "Pink Prism",
        "Quality": 145,
        "Bullet Resistance": 36.57,
        "Explosion Protection": 28.27,
        "Psy-Emissions": -2.5,
        "Stamina Regeneration": 6.79
    },
    {
        "name": "Pink Scallop",
        "Quality": 145,
        "Bullet Resistance": 25.45,
        "Explosion Protection": 19.6,
        "Radiation": -1.25,
        "Bleeding": -1.70
    },
    {
        "name": "Pink Frame",
        "Quality": 145,
        "Carry Weight": 23.37,
        "Stamina Regeneration": 5.66,
        "Radiation": 2.41,
        "Temperature": 2.41,
        "Psy-Emissions": 2.41,
        "Biological Infection": 2.41
    },
    {
        "name": "Pink Rime",
        "Quality": 145,
        "Bullet Resistance": 25.45,
        "Laceration Protection": 26.96,
        "Temperature": 0.36,
        "Frost": -0.5,
        "Explosion Protection": 14.70
    },
    {
        "name": "Pink Chilly",
        "Quality": 145,
        "Vitality": 6.79,
        "Temperature": 0.72,
        "Frost": -1,
        "Burning": -1.13,
        "Explosion Protection": 28.09
    },
    {
        "name": "Pink Heel",
        "Quality": 145,
        "Healing Effectiveness": 41.47,
        "Temperature": 0.72,
        "Frost": -1,
        "Reaction to laceration": 10.18,
        "Reaction to burns": 6.22
    },
    {
        "name": "Pink Firebird",
        "Quality": 145,
        "Health Regeneration": 5.09,
        "Healing Effectiveness": 27.71,
        "Temperature": -1.25,
        "Bleeding": -1.89
    },
    {
        "name": "Pink Viburnum Branch",
        "Quality": 145,
        "Vitality": 3.77,
        "Healing Effectiveness": 34.68,
        "Radiation": -2.5,
        "Stamina": 30.54
    },
    {
        "name": "Pink Transformer",
        "Quality": 145,
        "Stamina": 35.06,
        "Carry Weight": 16.40,
        "Biological Infection": 2.41,
        "Psy-Emissions": 5.73
    },
    {"name": "Pink Timber","Quality": 145,"Bullet Resistance": 20.36,"Radiation": 2.41,"Temperature": -0.63,"Health Regeneration": 3.77},
    {"name": "Pink Mug","Quality": 145,"Vitality": 3.02,"Radiation": -0.63,"Health Regeneration": 3.77},
    {"name": "Pink Snares","Quality": 145,"Psy-Emissions": 7.69,"Bullet Resistance": 12.82},
    {"name": "Pink Blackhole","Quality": 145,"Vitality": 1.32,"Health Regeneration": 9.05,"Bullet Resistance": 25.45,"Psy-Emissions": -2.5},
    {"name": "Pink Larva","Quality": 145,"Healing Effectiveness": 19.42,"Temperature": 2.41,"Biological Infection": -0.63,"Vitality": 2.26},
]

containers = [
    {"container": "Barrel",  "capacity": 7, "Internal Protection": 60, "Effectiveness": 93, "Frost": 0, "Healing Effectiveness": 0, "Psy-Emissions": 0},
    {"container": "Overton", "capacity": 6, "Internal Protection": 60, "Effectiveness": 100, "Healing Effectiveness": 31.3, "Frost": 0, "Psy-Emissions": 0},
    {"container": "BD6", "capacity": 6, "Internal Protection": 79, "Effectiveness": 100, "Frost": 0, "Healing Effectiveness": 0, "Psy-Emissions": 0},
    {"container": "Sheaf", "capacity": 7, "Internal Protection": 60, "Effectiveness": 97, "Frost": -1, "Healing Effectiveness": 0, "Psy-Emissions": 0},
    {"container": "Chitin", "capacity": 6, "Internal Protection": 60, "Effectiveness": 115, "Frost": 0, "Healing Effectiveness": 0, "Psy-Emissions": 0},
    {"container": "SMC", "capacity": 4, "Internal Protection": 95, "Effectiveness": 120, "Frost": 0, "Healing Effectiveness": 0, "Psy-Emissions": 0},
    {"container": "Z6", "capacity": 6, "Internal Protection": 75, "Effectiveness": 100, "Frost": 2.5, "Healing Effectiveness": 0, "Psy-Emissions": -0.5},
    {"container": "BD6u", "capacity": 6, "Internal Protection": 78.5, "Effectiveness": 114, "Frost": 0, "Healing Effectiveness": 0, "Psy-Emissions": 0},
]

armors = [
    {"armor": "Hector Exoskeleton", "Bullet Resistence": 312.26,"Periodic Healing": 0},
    {"armor": "CD4",                 "Bullet Resistence": 330.33,"Periodic Healing": 0},
    {"armor": "Alba Heavy", "Bullet Resistence": 340.81,"Periodic Healing": 0},
    {"armor": "SBA", "Bullet Resistence": 400.71,"Periodic Healing": 0},
    {"armor": "M2", "Bullet Resistence": 310.82, "Periodic Healing": 3},
    {"armor": "Cent", "Bullet Resistence": 355, "Periodic Healing": 0},
    {"armor": "Reiter", "Bullet Resistence": 433.07,"Periodic Healing": 0},
    {"armor": "Punisher", "Bullet Resistence": 258.9, "Periodic Healing": 0},
    {"armor": "RAPS", "Bullet Resistence": 297.88, "Periodic Healing": 2},
    {"armor": "Saturn", "Bullet Resistence": 261.48,"Periodic Healing": 0},
    {"armor": "JD ZIVCAS", "Bullet Resistence": 275.75,"Periodic Healing": 3},
    {"armor": "Rigel", "Bullet Resistence": 335.58, "Periodic Healing": 0}
]

weapons = [
    {"weapon": "AK-15", "damage": 64.88, "rof": 600},
    {"weapon": "QBZ", "damage": 51.75, "rof": 800},
    {"weapon": "X95 (A)", "damage": 44.13, "rof": 950}
]

medkit = [
    {"name": "Blue Medkit", "Periodic Healing": 6.0,  "Duration": 10.0},
    {"name": "STOMP",       "Periodic Healing": 12.6, "Duration": 5.0},
    {"name": "STRIKE",      "Periodic Healing": 10.0, "Duration": 5.0},
    {"name": "Green Medkit", "Periodic Healing": 7.0, "Duration": 12.0}
]
# ===================== NORMALIZE =====================
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))

def prepare_items_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and convert % → fractions for the correct fields."""
    df = df.copy()
    # Ensure columns
    needed = [
        "bullet_resistance","vitality","psy-emissions","radiation",
        "biological_infection","frost","temperature","healing_effectiveness",
        "periodic_healing","health_regeneration","name","quality"
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0

    # Convert percentage-like numeric columns to fractions, excluding absolute stats
    absolute_cols = {"name","bullet_resistance","quality","psy-emissions","radiation","biological_infection","frost","temperature"}
    num_cols = [c for c in df.columns if df[c].dtype != "O"]
    percent_like = [c for c in num_cols if c not in absolute_cols]
    df[percent_like] = df[percent_like].fillna(0.0) / 100.0
    return df

df_red_items  = prepare_items_df(norm_cols(pd.DataFrame(red_items)))
df_pink_items = prepare_items_df(norm_cols(pd.DataFrame(pink_items)))
df_containers = norm_cols(pd.DataFrame(containers))
df_armors     = norm_cols(pd.DataFrame(armors))
df_weapons    = norm_cols(pd.DataFrame(weapons))
df_medkits    = norm_cols(pd.DataFrame(medkit))

# unify armor bullet key
if "bullet_resistence" in df_armors.columns and "bullet_resistance" not in df_armors.columns:
    df_armors = df_armors.rename(columns={"bullet_resistence": "bullet_resistance"})

# ===================== CALC CORE =====================
def compositions(total: int, m: int):
    """Yield all nonnegative integer tuples of length m that sum to total."""
    if m == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, m - 1):
            yield (first,) + rest

def run_calc(
    armor_idx:int,
    container_idx:int,
    medkit_idx:int,
    weapon_idx:int,
    hit_frac:float,
    use_buffs:bool,
    use_limits:bool,
    df_items_override: Optional[pd.DataFrame] = None,
) -> str:
    """
    Returns a formatted string suitable for Discord.
    Pass df_items_override = df_red_items or df_pink_items from the bot.
    """
    # Pick item set
    df_items = df_items_override if df_items_override is not None else df_red_items

    chosen_armor     = df_armors.iloc[armor_idx]
    chosen_container = df_containers.iloc[container_idx]
    chosen_weapon    = df_weapons.iloc[weapon_idx]
    chosen_medkit    = df_medkits.iloc[medkit_idx]

    start_time = time.time()

    # Container params
    K   = int(chosen_container["capacity"])
    IP  = float(chosen_container.get("internal_protection", 0.0)) / 100.0  # info only
    EFF = float(chosen_container.get("effectiveness", 100.0)) / 100.0
    armor_bullet = float(chosen_armor.get("bullet_resistance", 0.0))

    # Container contributions (NOT scaled by EFF)
    def cont_get(key, as_fraction=False):
        val = chosen_container.get(key, 0.0)
        val = 0.0 if pd.isna(val) else float(val)
        return val/100.0 if as_fraction else val

    cont_psy    = cont_get("psy-emissions")
    cont_rad    = cont_get("radiation")
    cont_bio    = cont_get("biological_infection")
    cont_frost  = cont_get("frost")
    cont_temp   = cont_get("temperature")
    cont_heal   = cont_get("healing_effectiveness", as_fraction=True)  # fraction addend
    cont_ph_pct = cont_get("periodic_healing", as_fraction=True)       # fraction

    # Armor periodic healing (e.g., M2 has 3%)
    armor_ph_pct = float(chosen_armor.get("periodic_healing", 0.0)) / 100.0  # fraction

    # ===== Default buffs ===== (adjust to your latest values if needed)
    buff_vit      = 0.13  if use_buffs else 0.0   # +13% Vitality
    buff_he       = 0.321 if use_buffs else 0.0   # +32.1% Healing Effectiveness
    buff_hr       = 0.12  if use_buffs else 0.0   # +12% Health Regeneration
    buff_br_flat  = 9.9   if use_buffs else 0.0   # +9.9 flat Bullet Resistance

    # ===== Build arrays from chosen item set =====
    names  = df_items["name"].tolist()
    M      = len(names)

    bullet = df_items["bullet_resistance"].fillna(0.0).to_numpy(float)       # flat
    vital  = df_items["vitality"].fillna(0.0).to_numpy(float)                # fraction addend
    psy    = df_items["psy-emissions"].fillna(0.0).to_numpy(float)           # absolute
    rad    = df_items["radiation"].fillna(0.0).to_numpy(float)               # absolute
    bio    = df_items["biological_infection"].fillna(0.0).to_numpy(float)    # absolute
    frost  = df_items["frost"].fillna(0.0).to_numpy(float)                   # absolute
    temp   = df_items["temperature"].fillna(0.0).to_numpy(float)             # absolute
    heal   = df_items["healing_effectiveness"].fillna(0.0).to_numpy(float)   # fraction addend
    perh   = df_items["periodic_healing"].fillna(0.0).to_numpy(float)        # fraction
    hregen = df_items["health_regeneration"].fillna(0.0).to_numpy(float)     # fraction

    # Items scaled by container effectiveness
    adj_bullet = bullet * EFF
    adj_vital  = vital  * EFF
    adj_psy    = psy    * EFF
    adj_rad    = rad    * EFF
    adj_bio    = bio    * EFF
    adj_frost  = frost  * EFF
    adj_temp   = temp   * EFF
    adj_heal   = heal   * EFF
    adj_perh   = perh   * EFF
    adj_hreg   = hregen * EFF

    # Limits
    if use_limits:
        PSY_LIMIT, RAD_LIMIT, BIO_LIMIT, FROST_LIMIT, TEMP_LIMIT = -0.50, -0.50, -0.50, -1.00, -0.50
    else:
        PSY_LIMIT = RAD_LIMIT = BIO_LIMIT = FROST_LIMIT = TEMP_LIMIT = -9999

    # Incoming weapon stats
    weapon_name  = chosen_weapon["weapon"]
    weapon_dmg   = float(chosen_weapon["damage"])
    weapon_rpm   = float(chosen_weapon["rof"])
    shots_per_sec_eff = (weapon_rpm / 60.0) * hit_frac
    raw_dps       = weapon_dmg * shots_per_sec_eff

    # Search
    best = None  # (ttd, counts, total_bullet, vitality_mult, totals..., pipeline metrics)
    BASE_HEALTH_REGEN = 0.025  # 2.5% base HR

    for idx, counts in enumerate(compositions(K, M)):
        counts_arr = np.array(counts, dtype=float)

        # Totals for this combo
        items_bullet = float(np.dot(counts_arr, adj_bullet))
        total_bullet = armor_bullet + items_bullet + buff_br_flat  # include flat BR from buffs

        vitality_mult = 1.0 + float(np.dot(counts_arr, adj_vital)) + buff_vit
        DTK = (total_bullet + 100.0) * vitality_mult  # Effective health

        total_psy   = float(np.dot(counts_arr, adj_psy))   + cont_psy
        total_rad   = float(np.dot(counts_arr, adj_rad))   + cont_rad
        total_bio   = float(np.dot(counts_arr, adj_bio))   + cont_bio
        total_frost = float(np.dot(counts_arr, adj_frost)) + cont_frost
        total_temp  = float(np.dot(counts_arr, adj_temp))  + cont_temp

        # Caps
        if ((total_psy < PSY_LIMIT) or (total_rad < RAD_LIMIT) or
            (total_bio < BIO_LIMIT) or (total_frost < FROST_LIMIT) or
            (total_temp < TEMP_LIMIT)):
            continue

        # ---- DAMAGE PIPELINE ----
        D_raw = raw_dps

        # BR is represented via DTK; compute br_frac for report only
        br_frac = total_bullet / (total_bullet + 100.0) if total_bullet > -100.0 else 0.0

        # Healing Effectiveness & Health Regen
        total_HE = float(np.dot(counts_arr, adj_heal)) + cont_heal + buff_he
        total_HR = BASE_HEALTH_REGEN + float(np.dot(counts_arr, adj_hreg)) + buff_hr

        # Periodic Healing (as % DPS reduction), includes armor & medkit
        items_PH  = float(np.dot(counts_arr, adj_perh))
        cont_PH   = cont_ph_pct
        armor_PH  = armor_ph_pct
        medkit_PH = float(chosen_medkit["periodic_healing"]) / 100.0
        PH_total  = items_PH + cont_PH + armor_PH + medkit_PH

        # PH scales by (1 + HE + 0.20 * HR); clamp
        PH_effective_pct = PH_total * (1.0 + total_HE + 0.20 * total_HR)
        PH_effective_pct = max(0.0, min(PH_effective_pct, 0.99))

        # Final DPS after PH% reduction; BR already baked into DTK
        D_after_PH = D_raw * (1.0 - PH_effective_pct)
        net_dps = D_after_PH

        ttd = float('inf') if net_dps <= 0.0 else DTK / net_dps

        if (best is None) or (ttd > best[0]):
            best = (
                ttd, counts, total_bullet, vitality_mult,
                total_HE, total_HR, PH_total, PH_effective_pct,
                (total_psy, total_rad, total_bio, total_frost, total_temp),
                D_raw, D_after_PH, net_dps, br_frac, DTK, buff_br_flat,
                PSY_LIMIT, RAD_LIMIT, BIO_LIMIT, FROST_LIMIT, TEMP_LIMIT
            )

        # optional progress pulse
        if idx % 1000 == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(f"Checked {idx:,} combos in {elapsed:.1f} seconds...", end="\r")

    # ----- Build Output -----
    if best is None:
        return "\nNo valid combinations meet the psy/rad/bio/frost/temp constraints for this container."

    (ttd, counts, tb, vm,
     total_HE, total_HR, PH_total, PH_effective_pct,
     caps_tuple, D_raw, D_after_PH, net_dps, br_frac, DTK, buff_br_flat,
     PSY_LIMIT, RAD_LIMIT, BIO_LIMIT, FROST_LIMIT, TEMP_LIMIT) = best

    tot_psy, tot_rad, tot_bio, tot_frost, tot_temp = caps_tuple

    medkit_name = chosen_medkit["name"]
    weapon_name = chosen_weapon["weapon"]
    weapon_dmg  = float(chosen_weapon["damage"])
    weapon_rpm  = float(chosen_weapon["rof"])

    names  = df_items["name"].tolist()
    combo_list = []
    for n, nm in zip(counts, names):
        combo_list.extend([nm] * int(n))
    result_str = ", ".join(combo_list) if combo_list else "(no artifacts selected?)"

    lines = []
    lines.append(f"Use default buffs: {'YES' if use_buffs else 'NO'}")
    if use_buffs:
        lines.append("  Buffs → Vitality +13%, Healing Eff. +32.1%, Health Regen +12%, Bullet Resistance +9.9 (flat)")

    lines.append(f"\nArmor: {chosen_armor['armor']}")
    lines.append(f"Container: {chosen_container['container']} (capacity={int(chosen_container['capacity'])}, IP={IP*100:.0f}%, Effectiveness={EFF*100:.0f}%)")
    lines.append(f"Weapon: {weapon_name}  |  Damage {weapon_dmg:.2f}  |  RPM {weapon_rpm:.1f}  |  Hit% {hit_frac*100:.1f}%")
    lines.append(f"Medkit: {medkit_name}  |  Periodic Healing {float(chosen_medkit['periodic_healing']):.2f}%/s")

    lines.append("\nBest combination (max TTD):")
    lines.append(result_str)

    lines.append("\nBreakdown:")
    for n, nm in zip(counts, names):
        if n:
            lines.append(f"  {nm}: {n}")
    lines.append(f"  Armor bullet: {float(chosen_armor['bullet_resistance']):.2f}")
    lines.append(f"  Items bullet (after EFF): {tb - float(chosen_armor['bullet_resistance']) - buff_br_flat:.2f}")
    if use_buffs:
        lines.append(f"  + Flat BR from buffs: {buff_br_flat:.2f}")
    lines.append(f"  Total Bullet Resistance: {tb:.2f}")
    lines.append(f"  BR% from total BR (report only): {br_frac*100:.2f}%")
    lines.append(f"  Vitality multiplier: {vm:.4f}")

    lines.append("\nCaps:")
    lines.append(f"  Psy total: {tot_psy:.2f}   (limit: {PSY_LIMIT})")
    lines.append(f"  Rad total: {tot_rad:.2f}   (limit: {RAD_LIMIT})")
    lines.append(f"  Bio total: {tot_bio:.2f}   (limit: {BIO_LIMIT})")
    lines.append(f"  Frost total: {tot_frost:.2f} (limit: {FROST_LIMIT})")
    lines.append(f"  Temp total: {tot_temp:.2f}  (limit: {TEMP_LIMIT})")

    lines.append("\nHealing stats:")
    lines.append(f"  Healing Effectiveness (HE): {total_HE*100:.2f}%  (multiplier = {1+total_HE:.3f})")
    lines.append(f"  Health Regeneration (HR):   {total_HR*100:.2f}%")
    lines.append(f"  Periodic Healing (PH raw):  {PH_total*100:.2f}%")
    lines.append(f"  PH effective (DPS reduction): {PH_effective_pct*100:.2f}%")

    lines.append("\nDPS pipeline:")
    lines.append(f"  Raw DPS:          {D_raw:.2f}")
    lines.append(f"  After PH% reduce: {D_after_PH:.2f}")
    lines.append(f"  Net DPS:          {net_dps:.2f}")

    lines.append(f"\nDamage To Kill (DTK): {DTK:.2f}")
    lines.append(f"Time to die (TTD): {'INF' if np.isinf(ttd) else f'{ttd:.2f} s'}")

    elapsed = time.time() - start_time
    lines.append(f"\nDone! Search completed in {elapsed:.2f} seconds.")

    return "\n".join(lines)













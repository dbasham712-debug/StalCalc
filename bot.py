import os
import discord
from discord.ext import commands
from discord import app_commands

#choices
ARMOR_CHOICES = [
    app_commands.Choice(name="Hector Exoskeleton", value=0),
    app_commands.Choice(name="CD4", value=1),
    app_commands.Choice(name="Alba Heavy", value=2),
    app_commands.Choice(name="SBA", value=3),
    app_commands.Choice(name="M2", value=4),
    app_commands.Choice(name="Cent", value=5),
]
CONTAINER_CHOICES = [
    app_commands.Choice(name="Barrel", value=0),
    app_commands.Choice(name="Overton", value=1),
    app_commands.Choice(name="BD6", value=2),
    app_commands.Choice(name="Trailblazer", value=3),
    app_commands.Choice(name="Chitin", value=4),
    app_commands.Choice(name="SMC", value=5)
]
MEDKIT_CHOICES = [
    app_commands.Choice(name="Blue Medkit", value=0),
    app_commands.Choice(name="STOMP", value=1),
    app_commands.Choice(name="STRIKE", value=2),
    app_commands.Choice(name="Green Medkit", value=3)
]
WEAPON_CHOICES = [
    app_commands.Choice(name="AK-15", value=0),
    app_commands.Choice(name="QBZ", value=1),
]
# ---- bring in your data + calculator ----
from stalcalc_core import (
    df_armors, df_containers, df_medkits, df_weapons,
    run_calc
)
GUILD_ID = 1234

intents = discord.Intents.default()
client = commands.Bot(command_prefix="!", intents=intents)

@client.event
async def on_ready():
    try:
        guild = discord.Object(id=GUILD_ID)
        synced = await client.tree.sync(guild=guild)
        print(f"Synced {len(synced)} commands to guild {GUILD_ID}")
    except Exception as e:
        print(f"Failed to sync guild: {e}")
    print(f"Logged in as {client.user}")

# ---------- Helpers to render ID lists ----------
def fmt_id_list(df, label_col: str) -> str:
    lines = [f"ID: {i+1} = {label_col.title()}: {name}" for i, name in enumerate(df[label_col].tolist())]
    return "```\n" + "\n".join(lines) + "\n```"

# ---------- /armor ----------
@client.tree.command(name="armor", description="Get IDs for armor", guild=discord.Object(id=GUILD_ID))
async def armor_cmd(interaction: discord.Interaction):
    await interaction.response.send_message(fmt_id_list(df_armors, "armor"), ephemeral=True)

# ---------- /container ----------
@client.tree.command(name="container", description="Get IDs for containers", guild=discord.Object(id=GUILD_ID))
async def container_cmd(interaction: discord.Interaction):
    await interaction.response.send_message(fmt_id_list(df_containers, "container"), ephemeral=True)

# ---------- /medkit ----------
@client.tree.command(name="medkit", description="Get IDs for medkits", guild=discord.Object(id=GUILD_ID))
async def medkit_cmd(interaction: discord.Interaction):
    await interaction.response.send_message(fmt_id_list(df_medkits, "name"), ephemeral=True)

# ---------- /weapon ----------
@client.tree.command(name="weapon", description="Get IDs for weapons", guild=discord.Object(id=GUILD_ID))
async def weapon_cmd(interaction: discord.Interaction):
    await interaction.response.send_message(fmt_id_list(df_weapons, "weapon"), ephemeral=True)

# ---------- /calc ----------
@client.tree.command(name="calc", description="Calculate artifact build", guild=discord.Object(id=GUILD_ID))
@app_commands.describe(
    armor="Pick your armor",
    container="Pick your container",
    medkit="Pick your medkit",
    weapon="Pick your weapon",
    hit="Shooter hit % (0-100)",
    use_buffs="Use default buffs? (yes/no)",
    use_limits="Use limits for psy/rad/bio/frost/temp? (yes/no)"
)
@app_commands.choices(
    armor=ARMOR_CHOICES,
    container=CONTAINER_CHOICES,
    medkit=MEDKIT_CHOICES,
    weapon=WEAPON_CHOICES,
    use_buffs=[
        app_commands.Choice(name="yes", value="yes"),
        app_commands.Choice(name="no", value="no"),
    ],
    use_limits=[
        app_commands.Choice(name="yes", value="yes"),
        app_commands.Choice(name="no", value="no"),
    ]
)
async def calc_cmd(
    interaction: discord.Interaction,
    armor: app_commands.Choice[int],
    container: app_commands.Choice[int],
    medkit: app_commands.Choice[int],
    weapon: app_commands.Choice[int],
    hit: app_commands.Range[float, 0.0, 100.0],
    use_buffs: app_commands.Choice[str],
    use_limits: app_commands.Choice[str],
):
    await interaction.response.defer(thinking=True)

    armor_idx     = armor.value
    container_idx = container.value
    medkit_idx    = medkit.value
    weapon_idx    = weapon.value
    hit_frac      = hit / 100.0
    use_buffs_bool  = (use_buffs.value == "yes")
    use_limits_bool = (use_limits.value == "yes")

    try:
        output_text = run_calc(
            armor_idx=armor_idx,
            container_idx=container_idx,
            medkit_idx=medkit_idx,
            weapon_idx=weapon_idx,
            hit_frac=hit_frac,
            use_buffs=use_buffs_bool,
            use_limits=use_limits_bool,
        )
    except Exception as e:
        await interaction.followup.send(f"❌ Calculation error: `{e}`")
        return

    MAX = 1900
    if len(output_text) <= MAX:
        await interaction.followup.send(f"```\n{output_text}\n```")
    else:
        buf, count = [], 0
        for line in output_text.splitlines(True):
            if count + len(line) > MAX:
                await interaction.followup.send(f"```\n{''.join(buf)}\n```")
                buf, count = [], 0
            buf.append(line); count += len(line)
        if buf:
            await interaction.followup.send(f"```\n{''.join(buf)}\n```")

if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise SystemExit("Set DISCORD_TOKEN environment variable.")

client.run(TOKEN)

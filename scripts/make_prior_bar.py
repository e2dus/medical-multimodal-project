import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# match the paper font (AAAI / Times)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
})

labels = ["Demographic", "Random", "Zero", "Full\n(self prior)"]
pcc    = [0.652, 0.636, 0.651, 0.806]
colors = ["#aecbe8", "#aecbe8", "#aecbe8", "#1f5fd0"]  # light blue / strong blue

fig, ax = plt.subplots(figsize=(5.0, 3.2))
bars = ax.bar(labels, pcc, color=colors, width=0.6, edgecolor="none")
ax.tick_params(axis="x", labelsize=10)

ax.set_ylim(0.50, 0.85)
ax.set_ylabel(r"Mean PCC $\uparrow$")
ax.set_xlabel("Prior type")
ax.set_title("Prior-identity ablation")

for i, (b, v) in enumerate(zip(bars, pcc)):
    is_full = (i == len(bars) - 1)
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=12 if is_full else 9,
            fontweight="bold" if is_full else "normal",
            color="#1f5fd0" if is_full else "black")

# emphasize the Full (self prior) bar
bars[-1].set_edgecolor("#0b3a9c")
bars[-1].set_linewidth(1.6)

# bold the Full x-axis tick label
ax.get_xticklabels()[-1].set_fontweight("bold")
ax.get_xticklabels()[-1].set_color("#1f5fd0")

ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
# symmetric left/right margins so the plotting area sits centered
plt.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.20)
plt.savefig("prior_identity_bar.pdf")
plt.savefig("prior_identity_bar.png", dpi=160)
print("saved:", os.path.abspath("prior_identity_bar.pdf"))

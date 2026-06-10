"""Folium map generation for MarketMind recommendations."""
from pathlib import Path

import folium

from src.agents.data import filter_subset, load_features_df

RABAT_CENTER = [34.02, -6.83]


def _color(score: float) -> str:
    if score >= 70:
        return "green"
    if score >= 55:
        return "orange"
    return "red"


def build_map(district=None, btype=None) -> folium.Map:
    df = load_features_df()
    sub, _ = filter_subset(df, district, btype)

    if len(sub) == 0 or "latitude" not in sub.columns:
        return folium.Map(location=RABAT_CENTER, zoom_start=12, tiles="cartodbpositron")

    center = [sub["latitude"].mean(), sub["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    for _, row in sub.iterrows():
        score = float(row["success_score"])
        name = row.get("name", "?")
        rating = row.get("rating")
        popup = f"<b>{name}</b><br>Score: {score:.1f}"
        if rating == rating:  # not NaN
            popup += f"<br>Note: {rating}"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5 + score / 18,
            color=_color(score),
            fill=True,
            fill_color=_color(score),
            fill_opacity=0.85,
            weight=1,
            popup=folium.Popup(popup, max_width=200),
            tooltip=f"{name} ({score:.0f})",
        ).add_to(fmap)

    title = f"{btype or 'lieux'} — {district or 'Rabat'} (vert=fort, rouge=faible)"
    fmap.get_root().html.add_child(
        folium.Element(
            f'<div style="position:fixed;top:10px;left:50px;z-index:9999;'
            f'background:white;padding:6px 12px;border-radius:6px;'
            f'font-family:sans-serif;font-size:14px;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
            f"<b>{title}</b></div>"
        )
    )
    return fmap


def save_map(district=None, btype=None, out="data/processed/map.html") -> str:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    build_map(district, btype).save(out)
    return out


if __name__ == "__main__":
    print("Map saved to", save_map("Agdal", "cafe"))

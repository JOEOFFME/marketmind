"""Agent 2 — Data Retrieval (queries the feature data for the district/type)."""
from loguru import logger

from .data import filter_subset, load_features_df
from .state import AgentState


def retrieval_agent(state: AgentState) -> AgentState:
    parsed = state.get("parsed", {})
    district = parsed.get("district")
    btype = parsed.get("type")
    logger.info(f"[retrieval] district={district} type={btype}")

    df = load_features_df()
    sub, warnings = filter_subset(df, district, btype)
    result = {
        "district": district,
        "type": btype,
        "warnings": warnings,
        "n_places": int(len(sub)),
    }

    if len(sub):
        result["avg_success_score"] = round(float(sub["success_score"].mean()), 1)
        result["median_success_score"] = round(float(sub["success_score"].median()), 1)
        result["avg_rating"] = (
            round(float(sub["rating"].mean()), 2) if "rating" in sub.columns else None
        )
        top_cols = [
            c
            for c in ["name", "success_score", "rating", "latitude", "longitude"]
            if c in sub.columns
        ]
        result["top_places"] = (
            sub.nlargest(5, "success_score")[top_cols].to_dict(orient="records")
        )
        # all points for the map (name + coords + score)
        if "latitude" in sub.columns and "longitude" in sub.columns:
            pts_cols = [
                c
                for c in ["name", "latitude", "longitude", "success_score"]
                if c in sub.columns
            ]
            result["map_points"] = sub[pts_cols].to_dict(orient="records")

    dcol = f"district_{district}" if district else None
    if dcol and dcol in df.columns:
        result["district_avg_score_all_types"] = round(
            float(df[df[dcol] == 1]["success_score"].mean()), 1
        )

    logger.info(
        f"[retrieval] {result['n_places']} places, avg={result.get('avg_success_score')}"
    )
    return {"retrieved": result}


if __name__ == "__main__":
    import json

    from dotenv import load_dotenv

    from .query_agent import query_agent

    load_dotenv()
    s = {"question": "Où ouvrir un café à Agdal ?"}
    s.update(query_agent(s))
    s.update(retrieval_agent(s))
    r = dict(s["retrieved"])
    r["map_points"] = f"[{len(r.get('map_points', []))} points]"  # compact print
    print(json.dumps(r, indent=2, ensure_ascii=False))

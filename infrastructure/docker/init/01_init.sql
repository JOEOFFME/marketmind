-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Raw POIs from OpenStreetMap
CREATE TABLE IF NOT EXISTS raw_pois (
    id          SERIAL PRIMARY KEY,
    osm_id      BIGINT,
    name        TEXT,
    amenity     TEXT,
    shop        TEXT,
    landuse     TEXT,
    geometry    GEOMETRY(GEOMETRY, 4326),
    source      TEXT DEFAULT 'osm',
    collected_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_raw_pois_geometry ON raw_pois USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_raw_pois_amenity  ON raw_pois(amenity);

-- Marketplaces (curated from POIs)
CREATE TABLE IF NOT EXISTS marketplaces (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    district        TEXT,
    market_type     TEXT,  -- souk | mall | corridor | neighborhood | tourist
    geometry        GEOMETRY(POINT, 4326),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_marketplaces_geometry ON marketplaces USING GIST(geometry);

-- Weather observations
CREATE TABLE IF NOT EXISTS weather_observations (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    temp_max        FLOAT,
    temp_min        FLOAT,
    precipitation   FLOAT,
    wind_speed      FLOAT,
    collected_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_weather_date ON weather_observations(date);

COMMENT ON TABLE raw_pois         IS 'Raw OSM points of interest — Phase 1';
COMMENT ON TABLE marketplaces     IS 'Curated marketplace locations — Phase 1';
COMMENT ON TABLE weather_observations IS 'Daily weather for Rabat — Phase 1';

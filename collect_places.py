# collect_places.py  – run once per city
import overpy, csv, json
API  = overpy.Overpass()
# Delhi bounding box (SW lat, SW lon, NE lat, NE lon)
bbox = (28.40, 76.80, 28.90, 77.40)

QUERY = f"""
[out:json][timeout:180];
(
  node["tourism"~"museum|artwork|attraction|zoo|viewpoint"]( {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]} );
  node["historic"]( {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]} );
  node["amenity"~"restaurant|cafe|bar|pub"]( {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]} );
);
out center tags;
"""

result = API.query(QUERY)

with open("places_delhi.csv", "w", newline='', encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["osm_id","name","lat","lon","category","raw_tags"])
    for n in result.nodes:
        name = n.tags.get("name")
        if not name:                       # skip unnamed POIs
            continue
        cat  = n.tags.get("tourism") or n.tags.get("historic") or n.tags.get("amenity")
        w.writerow([n.id, name, n.lat, n.lon, cat, json.dumps(n.tags, ensure_ascii=False)])

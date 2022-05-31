import folium

center=(43.662643, -79.395689)
source_point = (43.664527, -79.392442)
destination_point = (43.659659, -79.397669)

m = folium.Map(location=center, zoom_start=15)
folium.Marker(location=source_point,icon=folium.Icon(color='red',icon='camera', prefix='fa')).add_to(m)
folium.Marker(location=center,icon=folium.Icon(color='blue',icon='graduation-cap', prefix='fa')).add_to(m)
folium.Marker(location=destination_point,icon=folium.Icon(color='green',icon='university', prefix='fa')).add_to(m)
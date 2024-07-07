// Initialize the map and set its view to Moscow
var map = L.map('map').setView([55.751244, 37.618423], 10);

// Add the OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

var markerObjects = [];  // To keep track of marker instances
var markerCoords = [];  // To keep track of marker coordinates
var districts = [];  // To keep track of district layers

// Function to add markers
map.on('click', function(e) {
    var marker = L.marker(e.latlng).addTo(map);
    marker.bindPopup("Marker at " + e.latlng.toString()).openPopup();
    markerObjects.push(marker);
    markerCoords.push([e.latlng.lat, e.latlng.lng]);
});

// Fetch and add Moscow districts polygons without adding markers
$.getJSON('/moscow_districts', function(data) {
    var geojson = osmtogeojson(data);
    L.geoJSON(geojson, {
        onEachFeature: function (feature, layer) {
            if (feature.properties && feature.properties.tags && feature.properties.tags.name) {
                layer.bindPopup(feature.properties.tags.name);
            }
            if (layer.hasOwnProperty("_bounds"))
            {
               districts.push(layer);
            }
        },
        pointToLayer: function(feature, latlng) {
            return L.circleMarker(latlng, {
                radius: 0,
                fillColor: "#ff7800",
                color: "#000",
                weight: 1,
                opacity: 0,
                fillOpacity: 0.8,
                transparency: 1
            });
        }
    }).addTo(map);
});

// Clear markers
document.getElementById('clearMarkers').onclick = function() {
    markerObjects.forEach(function(marker) {
        map.removeLayer(marker);
    });
    markerObjects = [];
    markerCoords = [];
};

// Export markers to CSV
document.getElementById('exportMarkers').onclick = function() {
    fetch('/export_markers', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(markerCoords)
    })
    .then(response => response.blob())
    .then(blob => {
        var url = window.URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'markers.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    })
    .catch(error => console.error('Error:', error));
};

// Import markers from CSV
document.getElementById('importMarkers').onclick = function() {
    document.getElementById('importMarkersFile').click();
};

document.getElementById('importMarkersFile').onchange = function() {
    var file = document.getElementById('importMarkersFile').files[0];
    var formData = new FormData();
    formData.append('file', file);

    fetch('/import_markers', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        data.forEach(function(coords) {
            var latlng = L.latLng(coords[0], coords[1]);
            var marker = L.marker(latlng).addTo(map);
            marker.bindPopup("Marker at " + latlng.toString()).openPopup();
            markerObjects.push(marker);
            markerCoords.push([latlng.lat, latlng.lng]);
        });
    })
    .catch(error => console.error('Error:', error));
};

// Enable import button only when a file is selected
//document.getElementById('importMarkersFile').addEventListener('change', function() {
//    document.getElementById('importMarkers').disabled = !this.files.length;
//});

// Fetch rate from ML model
document.getElementById('getRate').onclick = function() {
    var dropdown1 = document.getElementById('dropdown1').value;
    var dropdown2 = document.getElementById('dropdown2').value;
    var dropdown3 = document.getElementById('dropdown3').value;
    var dropdown4 = document.getElementById('dropdown4').value;

    fetch('/get_rate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            markers: markerCoords,
            dropdown1: dropdown1,
            dropdown2: dropdown2,
            dropdown3: dropdown3,
            dropdown4: dropdown4
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('rateResult').innerHTML = 'Rate: ' + data.rate;
    })
    .catch(error => console.error('Error:', error));
};

// Add random markers to the middle of N random polygons
document.getElementById('getOptimization').onclick = async function() {
    document.getElementById('optResult').innerHTML = "В процессе..."
    for (let i = 0; i<20; i++)
    {
        N = document.getElementById('banners').value;
        N = parseInt(N);

        var selectedDistricts = [];
        while (selectedDistricts.length < N && districts.length > 0) {
            var index = Math.floor(Math.random() * districts.length);
            selectedDistricts.push(districts.splice(index, 1)[0]);
        }
        console.log(selectedDistricts)
        console.log(districts.length)

        selectedDistricts.forEach(function(district) {
            console.log(district)
            var bounds = district.getBounds();
            var lat = (bounds.getNorth() + bounds.getSouth()) / 2;
            var lng = (bounds.getEast() + bounds.getWest()) / 2;
            var marker = L.marker([lat, lng]).addTo(map);
            marker.bindPopup("Marker at [" + lat + ", " + lng + "]").openPopup();
            markerObjects.push(marker);
            markerCoords.push([lat, lng]);
        });

            var dropdown1 = document.getElementById('dropdown1').value;
    var dropdown2 = document.getElementById('dropdown2').value;
    var dropdown3 = document.getElementById('dropdown3').value;
    var dropdown4 = document.getElementById('dropdown4').value;

    await fetch('/get_rate', {
        async: false,
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            markers: markerCoords,
            dropdown1: dropdown1,
            dropdown2: dropdown2,
            dropdown3: dropdown3,
            dropdown4: dropdown4
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('rateResult').innerHTML = 'Rate: ' + data.rate;
    })
    .catch(error => console.error('Error:', error));

            markerObjects.forEach(function(marker) {
        map.removeLayer(marker);
    });
    markerObjects = [];
    markerCoords = [];
    }
    document.getElementById('optResult').innerHTML = "Завершено"
};
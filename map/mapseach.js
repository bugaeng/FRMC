/*


let map; // 전역 변수로 map 선언

//gps 를끄면 검색기능이됨 ?


function initAutocomplete() {
    const map = new google.maps.Map(document.getElementById("map"), {
      center: { lat: -33.8688, lng: 151.2195 },
      zoom: 13,
      mapTypeId: "roadmap",
    });

    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
          const initialLocation = new google.maps.LatLng(
            position.coords.latitude,
            position.coords.longitude
          );
          map.setCenter(initialLocation);
        });
      }


  
  const input = document.getElementById("pac-input");
input.addEventListener("keydown", function(e) {
  if (e.keyCode === 13) {
    e.preventDefault();
    const searchBox = new google.maps.places.SearchBox(input);
    const places = searchBox.getPlaces();
    if (places && places.length > 0) {
      const location = places[0].geometry.location;
      map.panTo(location);
    }
  }
});


  const searchBox = new google.maps.places.SearchBox(input);

  map.controls[google.maps.ControlPosition.TOP_LEFT].push(input);
  // Bias the SearchBox results towards current map's viewport.
  map.addListener("bounds_changed", () => {
    searchBox.setBounds(map.getBounds());
  });

  let markers = [];

  // Listen for the event fired when the user selects a prediction and retrieve
  // more details for that place.
  searchBox.addListener("places_changed", () => {
    const places = searchBox.getPlaces();

    if (places.length == 0) {
      return;
    }

    // Clear out the old markers.
    markers.forEach((marker) => {
      marker.setMap(null);
    });
    markers = [];

    // For each place, get the icon, name and location.
    const bounds = new google.maps.LatLngBounds();

    places.forEach((place) => {
      if (!place.geometry || !place.geometry.location) {
        console.log("Returned place contains no geometry");
        return;
      }

      const icon = {
        url: place.icon,
        size: new google.maps.Size(71, 71),
        origin: new google.maps.Point(0, 0),
        anchor: new google.maps.Point(17, 34),
        scaledSize: new google.maps.Size(25, 25),
      };

      // Create a marker for each place.
      markers.push(
        new google.maps.Marker({
          map,
          icon,
          title: place.name,
          position: place.geometry.location,
        }),
      );
      if (place.geometry.viewport) {
        // Only geocodes have viewport.
        bounds.union(place.geometry.viewport);
      } else {
        bounds.extend(place.geometry.location);
      }
    });
    map.fitBounds(bounds);
  });



  
}

const input = document.getElementById("pac-input");
input.addEventListener("keydown", function(e) {
  if (e.keyCode === 13) { // 엔터 키 코드
    e.preventDefault(); // 기본 동작 방지
    const searchBox = new google.maps.places.SearchBox(input);
    const places = searchBox.getPlaces();
    if (places && places.length > 0) {
      const location = places[0].geometry.location;
      map.panTo(location); // 검색 결과로 이동
    }
  }
});

window.initAutocomplete = initAutocomplete;
*/
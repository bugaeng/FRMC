


//밑에 로컬값을 수정 해야 안면에따라서  

const saveLabel = localStorage.getItem("saveLabel");

function showPosition(position) {
  var myLocation = {
    lat: position.coords.latitude,
    lng: position.coords.longitude,
    
  };
  



  var map = new google.maps.Map(document.getElementById('map'), {
    zoom: 14,
    center: myLocation, 
    

  });
  
  const onChangeHandler = function () {
    calculateAndDisplayRoute(directionsService, directionsRenderer);
  };


  
  var marker = new google.maps.Marker({
    position: myLocation,
    map: map,
    title: '자신의 위치',
    label: "내위치",
    animation: google.maps.Animation.DROP,
    panControl: true,
    zoomControl: true,
    mapTypeControl: true,
    scaleControl: true,
    streetViewControl: true,
    overviewMapControl: true,
    rotateControl: true

    
  });

  let storetotal = localStorage.getItem("total-value")
  let storeskinlabel = localStorage.getItem("skin_label")
  //합칠때 saveLabel2 수정바람 
  let malls = [];
 

  //얘는 조건 값 하나
 
  //얘는  조건값 둘
if (storeskinlabel != 'Normal' && storetotal > 10)  { 
      malls = [
          { label: "대전한국병원", lat: 36.3482, lng: 127.4358 , text: "대전한국병원" ,  callnumber:"042-606-1000",link: "https://www.google.com/maps/place/%EB%8C%80%EC%A0%84%EB%B3%91%EC%9B%90/data=!4m10!1m2!2m1!1z64yA7KCE67OR7JuQ!3m6!1s0x356549aee7152755:0xda227fb0adb82722!8m2!3d36.3625535!4d127.417779!15sCgzrjIDsoITrs5Hsm5CSAQ5tZWRpY2FsX2NlbnRlcuABAA!16s%2Fg%2F1xpwgn0_?hl=ko&entry=ttu"},
          { label: "을지대학교병원", lat: 36.3553303, lng: 127.3820305 , text:"을지대학교병원" ,  callnumber:"1899-0001", link: "https://www.emc.ac.kr/"  },
          { label: "대청병원", lat: 36.3083034, lng: 127.3703807 , text:"대청병원", callnumber:"042-1899-6075", link: "http://www.dchp.or.kr/"},
          { label: "건양대병원", lat: 36.3066294, lng: 127.3424035 , text:"건양대학교", callnumber:"041-730-5114", link: "https://www.kyuh.ac.kr/"},
          { label: "대전선병원", lat: 36.3361, lng: 127.4107 , text:"대전선병원", callnumber:"1588-7011", link: "https://www.sunhospital.com/index.html"},
          { label: "대전성모병원", lat: 36.3222, lng: 127.4205 , text:"대전성모병원", callnumber:"1577-0888", link: "https://www.cmcdj.or.kr/index.jsp?_MOBILE_PC=Y"},
          { label: "충남대학병원", lat: 36.3168, lng: 127.4162 , text:"충남대학병원", callnumber:"1599-7123", link: "https://www.cnuh.co.kr/home/index.do"},
          { label: "유성선병원", lat: 36.3752, lng: 127.3248 , text:"유성선병원", callnumber:"1588-7011(ARS 3번)", link: "https://www.yuseongsunhospital.com/index.html"}
      ];
  }
  else if (storeskinlabel != 'Normal' && storetotal <= 10) {
    
    malls = [
      { label: "킴벨피부과", lat: 36.354, lng: 127.3799 , text:"킴벨피부과", callnumber: "042-471-7575",link: "http://www.kimbelle.kr/"},
      { label: "엠제이피부과", lat: 36.3494, lng: 127.3786 , text:"엠제이피부과", callnumber: "042-471-0880",link: "http://www.mjskin.co.kr/"},
      { label: "폴라인피부과", lat: 36.3506, lng: 127.3898 , text:"폴라인피부과", callnumber: "042-476-0002",link: "http://www.paulline.co.kr/default/"},
      { label: "청담 피부과", lat: 36.3527, lng: 127.3773 , text:"청담 피부과", callnumber: "	042-471-5300",link: "http://www.cddc.co.kr/"},
      { label: "조아피부과의원", lat: 36.3745, lng: 127.3184 , text:"조아피부과의원", callnumber: "042-822-2075",link: "http://www.joaskin.co.kr/"},
      { label: "이종성피부과의원", lat: 36.3127, lng: 127.3803 , text:"이종성피부과의원", callnumber: "042-585-1402",link: ""},
      { label: "영피부과", lat: 36.3675, lng: 127.4294 , text:"영피부과", callnumber: "042-637-7755",link: ""},
      { label: "럭스피부과의원", lat: 36.3756, lng: 127.3186 , text:"럭스피부과의원", callnumber: "042-485-1112",link: "https://www.luxskin.co.kr/"},

    ];
}

    //여기서 신경외과 추천 해주기 얘는하나 
 else if (storeskinlabel === 'Normal'  && storetotal > 10) {
      malls = [
          { label: "위캔두 신경과", lat: 36.3594, lng: 127.3784 , text: "위캔두신경과" , callnumber:"042-487-0011", link: "https://wecando.kr/"},
          { label: "선사신경외과의원", lat: 36.3587, lng: 127.3769 , text:"선사신경외과의원" , callnumber : "042-472-9975" ,link:""  }, 
          { label: "DH안광병신경과", lat: 36.3545, lng: 127.38 , text:"DH안광병신경과", callnumber: "042-486-8100", link: "https://dh-neurologist.business.site/?utm_source=gmb&utm_medium=referral"},
          { label: "대전우리병원", lat: 36.3436, lng: 127.3838 , text:"대전우리병원", callnumber: "1577-0052",link: "https://www.woorispine.com/default/"},
          { label: "좋은날신경과의원", lat: 36.299, lng: 127.3246 , text:"좋은날신경과의원", callnumber: "042-542-3588",link: "https://www.gooddayclinic.co.kr/"},
          { label: "비엔피병원", lat: 36.3544, lng: 127.3399 , text:"비엔피병원", callnumber: "042-522-8275",link: "http://bnphospital.co.kr/"},
          { label: "브레인업신경과의원", lat: 36.3515, lng: 127.3874 , text:"브레인업신경과의원", callnumber: "042-484-3330",link: "https://brainupclinic.modoo.at/"},
          { label: "한동균신경과", lat: 36.3485, lng: 127.4343 , text:"한동균신경과", callnumber: "042-710-7582",link: ""},

      ];
  }







/********************************* */
 
// 킴벨피부과 아이콘
const iconUrl1 = "https://cdn-icons-png.flaticon.com/512/3755/3755512.png";
// 엠제이피부과 아이콘
const iconUrl2 = "https://cdn-icons-png.flaticon.com/512/3755/3755512.png";
// 폴라인피부과 아이콘
const iconUrl3 = "https://cdn-icons-png.flaticon.com/512/3755/3755512.png";
// 청담 피부과 아이콘
const iconUrl4 = "https://cdn-icons-png.flaticon.com/512/3755/3755512.png";
// 조아피부과의원 아이콘
const iconUrl5 = "https://cdn-icons-png.flaticon.com/512/3755/3755512.png";
// 이종성피부과의원 아이콘
const iconUrl6 = "https://cdn-icons-png.flaticon.com/512/3755/3755512.png";
// 영피부과 아이콘
const iconUrl7 = "https://cdn-icons-png.flaticon.com/512/3755/3755512.png";
// 럭스피부과의원 아이콘
const iconUrl8 = "https://cdn-icons-png.flaticon.com/512/3755/3755512.png";

// 여기서부터 대학병원 

const iconUrl9 = "https://cdn-icons-png.flaticon.com/512/376/376433.png";
const iconUrl10 = "https://cdn-icons-png.flaticon.com/512/376/376433.png";
const iconUrl11 = "https://cdn-icons-png.flaticon.com/512/376/376433.png";
const iconUrl12 = "https://cdn-icons-png.flaticon.com/512/376/376433.png";
const iconUrl13 = "https://cdn-icons-png.flaticon.com/512/376/376433.png";
const iconUrl14 = "https://cdn-icons-png.flaticon.com/512/376/376433.png";
const iconUrl15 = "https://cdn-icons-png.flaticon.com/512/376/376433.png";
const iconUrl16 = "https://cdn-icons-png.flaticon.com/512/376/376433.png";


//여기서 부터 신경과 

const iconUrl17 = "https://cdn-icons-png.flaticon.com/512/2317/2317969.png";
const iconUrl18 = "https://cdn-icons-png.flaticon.com/512/2317/2317969.png";
const iconUrl19 = "https://cdn-icons-png.flaticon.com/512/2317/2317969.png";
const iconUrl20 = "https://cdn-icons-png.flaticon.com/512/2317/2317969.png";
const iconUrl21 = "https://cdn-icons-png.flaticon.com/512/2317/2317969.png";
const iconUrl22 = "https://cdn-icons-png.flaticon.com/512/2317/2317969.png";
const iconUrl23 = "https://cdn-icons-png.flaticon.com/512/2317/2317969.png";
const iconUrl24 = "https://cdn-icons-png.flaticon.com/512/2317/2317969.png";




//

malls.forEach(({ label, lat, lng, text, callnumber, link }, index) => {
  let iconUrl = "";

  if (label === "킴벨피부과") {
    iconUrl = iconUrl1;
  } else if (label === "엠제이피부과") {
    iconUrl = iconUrl2;
  } else if (label === "폴라인피부과") {
    iconUrl = iconUrl3;
  } else if (label === "청담 피부과") {
    iconUrl = iconUrl4;
  } else if (label === "조아피부과의원") {
    iconUrl = iconUrl5;
  } else if (label === "이종성피부과의원") {
    iconUrl = iconUrl6;
  } else if (label === "영피부과") {
    iconUrl = iconUrl7;
  } else if (label === "럭스피부과의원") {
    iconUrl = iconUrl8;
  } 

  if (label === "대전한국병원") {
    iconUrl = iconUrl9;
  } else if (label === "을지대학교병원") {
    iconUrl = iconUrl10;
  } else if (label === "대청병원") {
    iconUrl = iconUrl11;
  } else if (label === "건양대병원") {
    iconUrl = iconUrl12;
  } else if (label === "대전선병원") {
    iconUrl = iconUrl13;
  } else if (label === "대전성모병원") {
    iconUrl = iconUrl14;
  } else if (label === "충남대학병원") {
    iconUrl = iconUrl15;
  } else if (label === "유성선병원") {
    iconUrl = iconUrl16;
  } 

  if (label === "위캔두 신경과") {
    iconUrl = iconUrl17;
  } else if (label === "선사신경외과의원") {
    iconUrl = iconUrl18;
  } else if (label === "DH안광병신경과") {
    iconUrl = iconUrl19;
  } else if (label === "대전우리병원") {
    iconUrl = iconUrl20;
  } else if (label === "좋은날신경과의원") {
    iconUrl = iconUrl21;
  } else if (label === "비엔피병원") {
    iconUrl = iconUrl22;
  } else if (label === "브레인업신경과의원") {
    iconUrl = iconUrl23;
  } else if (label === "한동균신경과") {
    iconUrl = iconUrl24;
  }

  const marker = new google.maps.Marker({
    position: { lat, lng },
    label: {
      text: label,
      callnumber: callnumber,
      color: "black",
      fontWeight: "normal",
      fontSize: "25px"
    },
    icon: {
      url: iconUrl,
      scaledSize: new google.maps.Size(60, 60),
      labelOrigin: new google.maps.Point(0, -14)
    },
    
    animation: google.maps.Animation.DROP,
    map
 
  });


    const content = `<div>
      <h3>${label}</h3>
      <p>${text || ""}</p> 

      <p>${callnumber || ""}</p>
      <br>
      <a href="${link}" target="_blank">${link}</a>
    </div>`;

    google.maps.event.addListener(marker, 'click', function() {
      const infoWindow = new google.maps.InfoWindow({
        
        content: content
      });
      infoWindow.open(map, marker);
      map.setZoom(18); //클릭시 줌 합니다.
      map.setCenter(marker.getPosition());

     
    });
  });






   
  const searchInput = document.getElementById("search-input");

  const searchBox = new google.maps.places.SearchBox(searchInput);
  
  searchBox.addListener("places_changed", () => {
    const places = searchBox.getPlaces();
    if (places.length === 0) {
      return;
    }



    

    const bounds = new google.maps.LatLngBounds();
    places.forEach((place) => {
      if (!place.geometry || !place.geometry.location) {
        console.log("Returned place contains no geometry");
        return;
      }

    

      google.maps.event.addListener(marker, "click", function () {
        const infoWindow = new google.maps.InfoWindow({
          content: `
            <div>
              <h3>${place.name}</h3>
              <p>${place.formatted_address}</p>
              <a href="${place.website}" target="_blank">${place.website}</a>
            </div>
          `,
        });
        infoWindow.open(map, marker);
      });

      if (place.geometry.viewport) {
        bounds.union(place.geometry.viewport);
      } else {
        bounds.extend(place.geometry.location);
      }
    });

    map.fitBounds(bounds);
  });
}

if (navigator.geolocation) {
  navigator.geolocation.getCurrentPosition(showPosition);
} else {
  alert('Geolocation is not supported by this browser.');
}



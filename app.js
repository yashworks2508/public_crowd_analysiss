const appData = {
  routes: [
    { id: "R1", name: "Blue Line", type: "Rail", capacity: 200 },
    { id: "B15", name: "Bus 15", type: "Bus", capacity: 50 },
    { id: "B42", name: "Bus 42", type: "Bus", capacity: 50 }
  ],
  stops: [
    { id: "S1", name: "Downtown Station", major: true },
    { id: "S2", name: "University Plaza", major: false },
    { id: "S3", name: "Stadium", major: true },
    { id: "S4", name: "Shopping Mall", major: false },
    { id: "S5", name: "Business District", major: true },
    { id: "S6", name: "Airport Terminal", major: true }
  ],
  weather: {
    temperature: 72,
    condition: "Clear",
    impact: "Normal ridership expected"
  },
  events: [
    { name: "Baseball Game", location: "Stadium" },
    { name: "Concert", location: "Downtown Station" }
  ]
};

function updateTimestamp() {
  document.getElementById("timestamp").textContent = new Date().toLocaleString();
}
setInterval(updateTimestamp, 1000);
updateTimestamp();

async function getPrediction(features) {
  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features)
    });
    return await response.json();
  } catch (err) {
    console.error('Error fetching prediction:', err);
    return null;
  }
}

function getCrowdingClass(level) {
  if (level === 'Low') return 'crowdingLow';
  else if (level === 'Medium') return 'crowdingMedium';
  else return 'crowdingHigh';
}

async function renderRouteGrid() {
  const grid = document.getElementById("routeGrid");
  grid.innerHTML = '';

  const hour = new Date().getHours();
  const dayOfWeek = new Date().getDay() - 1; // Sunday=0, convert to Monday=0
  const isWeekend = dayOfWeek >= 5 ? 1 : 0;
  const weatherNumeric = { 'Clear': 1, 'Cloudy': 2, 'Rain': 3 }[appData.weather.condition] || 1;
  const featureTemplate = {
    hour,
    day_of_week: dayOfWeek,
    is_weekend: isWeekend,
    temperature: appData.weather.temperature,
    weather_numeric: weatherNumeric,
    event_impact: 1.0,
    has_event: 0,
    is_morning_rush: (hour >= 7 && hour <= 9) ? 1 : 0,
    is_evening_rush: (hour >= 17 && hour <= 19) ? 1 : 0,
    major_stop: 0,
    route_type_encoded: 0,
    vehicle_capacity: 0
  };

  for (const route of appData.routes) {
    for (const stop of appData.stops) {
      let features = {...featureTemplate};
      features.route_type_encoded = (route.type === 'Rail') ? 1 : 0;
      features.vehicle_capacity = route.capacity;
      features.major_stop = stop.major ? 1 : 0;

      // Check if event affects this stop
      const eventHere = appData.events.find(e => e.location === stop.name);
      if(eventHere){
        features.event_impact = 1.8;
        features.has_event = 1;
      }

      const prediction = await getPrediction(features);
      if(!prediction){
        console.error('Failed prediction for', route.name, stop.name);
        continue;
      }

      const crowdClass = getCrowdingClass(prediction.crowd_level);

      const card = document.createElement('div');
      card.className = `route-card ${crowdClass}`;
      card.innerHTML = `
        <h3>${route.name} - ${stop.name}</h3>
        <p><strong>Passengers:</strong> ${prediction.predicted_passengers} / ${route.capacity}</p>
        <p><strong>Crowding Level:</strong> ${prediction.crowd_level} (${(prediction.predicted_crowding_ratio * 100).toFixed(0)}%)</p>
      `;
      grid.appendChild(card);
    }
  }
}

renderRouteGrid();
setInterval(renderRouteGrid, 30000);
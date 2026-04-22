from flask import Flask, request, jsonify, render_template_string
from bs4 import BeautifulSoup
from groq import Groq
import requests
import json
import os
import re
from datetime import datetime

app = Flask(__name__)
GROQ_KEY = "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"
FEEDBACK_FILE = "C:/Users/vedan/WebScrapingAgent/feedback_data.json"

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Solar Plant AI Planner</title>
    <style>
        *{box-sizing:border-box;margin:0;padding:0;font-family:Arial}
        body{background:#f0f8ff;padding:20px}
        .container{max-width:950px;margin:0 auto}
        h1{color:#e47911;text-align:center;padding:20px 0;font-size:28px}
        p.sub{text-align:center;color:#666;margin-bottom:20px}
        .form-card{background:white;padding:25px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);margin-bottom:20px}
        .grid2{display:grid;grid-template-columns:1fr 1fr;gap:15px}
        .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px}
        label{font-size:13px;color:#555;display:block;margin-bottom:5px;font-weight:bold}
        input,select{width:100%;padding:10px;border:1px solid #ddd;border-radius:8px;font-size:14px}
        input[readonly]{background:#f0f8ff;color:#006994;font-weight:bold}
        .btn{background:#f5a623;color:white;padding:14px;border:none;border-radius:8px;cursor:pointer;width:100%;font-size:16px;margin-top:15px;font-weight:bold}
        .btn:hover{background:#e09612}
        .btn2{background:#006994;color:white;padding:14px;border:none;border-radius:8px;cursor:pointer;width:100%;font-size:16px;margin-top:10px;font-weight:bold}
        .btn2:hover{background:#005580}
        .input-box{background:#f0f8ff;padding:15px;border-radius:8px;border:2px dashed #006994;margin-bottom:15px}
        .input-box h3{color:#006994;margin-bottom:10px;font-size:15px}
        .or-divider{text-align:center;color:#999;font-size:13px;margin:10px 0}
        .result{background:white;padding:25px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);display:none;margin-top:20px}
        .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
        .metric{background:#fff8f0;padding:15px;border-radius:10px;text-align:center;border:1px solid #f5a623}
        .metric .val{font-size:22px;font-weight:bold;color:#e47911}
        .metric .lbl{font-size:12px;color:#888;margin-top:4px}
        .section{margin-bottom:20px}
        .section h3{color:#006994;font-size:16px;margin-bottom:10px;padding-bottom:6px;border-bottom:2px solid #f0f8ff}
        .item{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f5f5f5;font-size:14px}
        .item-name{color:#555}
        .item-val{font-weight:bold;color:#333}
        .total{background:#006994;color:white;padding:12px;border-radius:8px;display:flex;justify-content:space-between;font-size:16px;font-weight:bold;margin-top:10px}
        .ai-box{background:#f0f8ff;padding:20px;border-radius:10px;border-left:4px solid #006994;font-size:14px;line-height:1.8;white-space:pre-wrap}
        .weather-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:15px}
        .w-card{background:#fff8f0;padding:12px;border-radius:8px;text-align:center;border:1px solid #f5a623}
        .w-card .wval{font-size:18px;font-weight:bold;color:#e47911}
        .w-card .wlbl{font-size:11px;color:#888;margin-top:3px}
        .tag{background:#e1f5ee;color:#0f6e56;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:bold}
        .warning{background:#fff3cd;padding:12px;border-radius:8px;font-size:13px;color:#856404;margin-top:10px}
        iframe{border:none;border-radius:10px;margin-top:10px;width:100%;height:350px}
        .coord-display{font-size:13px;color:#006994;margin-top:8px;font-weight:bold;padding:8px;background:#e8f4f8;border-radius:6px}
        .sun-badge{background:#e47911;color:white;padding:8px 15px;border-radius:20px;font-size:13px;font-weight:bold;display:inline-block;margin:4px}
        .nasa-badge{background:#003087;color:white;padding:8px 15px;border-radius:20px;font-size:13px;display:inline-block;margin:4px}
        .pvgis-badge{background:#2e7d32;color:white;padding:8px 15px;border-radius:20px;font-size:13px;display:inline-block;margin:4px}
        .kenbrook-badge{background:#c62828;color:white;padding:8px 15px;border-radius:20px;font-size:13px;display:inline-block;margin:4px}
        .elec-badge{background:#6a1b9a;color:white;padding:8px 15px;border-radius:20px;font-size:13px;display:inline-block;margin:4px}
        .source-box{background:#f8f9fa;padding:12px;border-radius:8px;margin-bottom:12px;font-size:13px;border-left:4px solid #006994}
        .status{background:#e8f5e9;padding:10px;border-radius:6px;font-size:13px;color:#2e7d32;margin-top:8px}
        .feedback-box{background:#f8f9fa;padding:15px;border-radius:8px;margin-top:15px;border:1px solid #ddd;display:none}
        .feedback-box p{font-size:14px;color:#555;margin-bottom:10px}
        .btn-correct{background:#28a745;color:white;padding:8px 20px;border:none;border-radius:6px;cursor:pointer;margin-right:10px;font-size:14px}
        .btn-wrong{background:#dc3545;color:white;padding:8px 20px;border:none;border-radius:6px;cursor:pointer;margin-right:10px;font-size:14px}
        .correction-input{width:320px;padding:8px;border:1px solid #ddd;border-radius:6px;font-size:13px;margin-top:8px}
        .feedback-stats{background:#e8f4f8;padding:10px;border-radius:6px;font-size:13px;color:#006994;margin-top:10px}
    </style>
</head>
<body>
<div class="container">
    <h1>☀️ Solar Plant AI Planner</h1>
    <p class="sub">NASA + PVGIS + Kenbrook + Auto Electricity Rate + AI Learning System!</p>

    <div class="form-card">
        <div class="input-box">
            <h3>📍 Enter Location (Coordinates OR Name)</h3>
            <div class="grid2">
                <div>
                    <label>🌐 Latitude (optional if name given)</label>
                    <input type="number" id="lat" placeholder="e.g. 17.9784" step="0.0001"/>
                </div>
                <div>
                    <label>🌐 Longitude (optional if name given)</label>
                    <input type="number" id="lon" placeholder="e.g. 79.5300" step="0.0001"/>
                </div>
            </div>
            <div class="or-divider">— OR —</div>
            <div>
                <label>🔍 Location Name (AI finds coordinates automatically)</label>
                <input type="text" id="locName" placeholder="e.g. NIT Warangal, Telangana"/>
            </div>
            <div class="coord-display" id="coord_display">📍 Enter coordinates or location name above</div>
        </div>

        <div class="grid3">
            <div>
                <label>📐 Area (Square Meters)</label>
                <input type="number" id="area" value="1000" min="100"/>
            </div>
            <div>
                <label>⚡ Panel Type</label>
                <select id="paneltype">
                    <option value="Monocrystalline">Monocrystalline (Best)</option>
                    <option value="Polycrystalline">Polycrystalline (Standard)</option>
                    <option value="Thin Film">Thin Film (Budget)</option>
                </select>
            </div>
            <div>
                <label>🏗️ Project Type</label>
                <select id="projtype">
                    <option value="Residential">Residential</option>
                    <option value="Commercial">Commercial</option>
                    <option value="Industrial">Industrial</option>
                    <option value="Agricultural">Agricultural</option>
                </select>
            </div>
        </div>
        <div style="background:#e8f4f8;padding:10px;border-radius:6px;margin-top:10px;font-size:13px;color:#006994">
            ⚡ Electricity rate will be auto-scraped from your location — no manual input needed!
        </div>
        <button class="btn2" onclick="getAIReport()">🤖 Scrape All Sources + AI Report + Map</button>
    </div>

    <div class="result" id="result">
        <div class="status" id="status_box"></div>
        <div class="metrics" id="metrics" style="margin-top:15px"></div>

        <div class="section" id="map_section" style="display:none">
            <h3>🗺️ Your Land on Map</h3>
            <iframe id="map_frame"></iframe>
            <h3 style="margin-top:15px">🛰️ Multi-Source Scraped Data</h3>
            <div id="source_info"></div>
            <div class="weather-grid" id="weather_box"></div>
        </div>

        <div class="section">
            <h3>🔧 Equipment Required (Kenbrook Solar Prices)</h3>
            <div id="equipment"></div>
        </div>
        <div class="section">
            <h3>💰 Cost Breakdown</h3>
            <div id="costs"></div>
        </div>
        <div class="section">
            <h3>📈 ROI & Recovery</h3>
            <div id="roi"></div>
        </div>
        <div class="section">
            <h3>🤖 AI Analysis</h3>
            <div class="ai-box" id="ai_answer">Analyzing...</div>
        </div>

        <div class="feedback-box" id="feedback_box">
            <p>🎯 Was this prediction accurate? Help the AI learn!</p>
            <button class="btn-correct" onclick="sendFeedback(1)">👍 Correct Prediction</button>
            <button class="btn-wrong" onclick="sendFeedback(-1)">👎 Wrong Prediction</button>
            <br>
            <input type="text" class="correction-input" id="correction"
                placeholder="What was actual PSH / ROI / rate? (optional correction)"/>
            <div class="feedback-stats" id="feedback_stats"></div>
        </div>

        <div class="warning">⚠️ Prices based on Kenbrook Solar scraping. Always get 3 quotes.</div>
    </div>
</div>

<script>
let annualPSH = 5.0;
let panelPrice = 18000;
let inverterPrice = 8000;
let elecRate = 7.0;
let lastPrediction = {};

async function getAIReport(){
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;
    const locName = document.getElementById('locName').value;
    const area = document.getElementById('area').value;
    const panelType = document.getElementById('paneltype').value;
    const projtype = document.getElementById('projtype').value;

    if(!lat && !lon && !locName){
        alert('Please enter coordinates OR location name!');
        return;
    }

    document.getElementById('result').style.display='block';
    document.getElementById('status_box').innerHTML='⏳ Finding coordinates... scraping NASA + PVGIS + Kenbrook + electricity rate...';
    document.getElementById('ai_answer').innerHTML='⏳ Scraping all data sources... Please wait 60 seconds...';
    document.getElementById('feedback_box').style.display='none';

    try{
        const r = await fetch('/analyze',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({lat,lon,locName,area,panelType,projtype})
        });
        const d = await r.json();

        if(d.error){
            document.getElementById('ai_answer').innerHTML='Error: '+d.error;
            return;
        }

        annualPSH = d.map.annual_psh || 5.0;
        panelPrice = d.prices.panel_price || 18000;
        inverterPrice = d.prices.inverter_price || 8000;
        elecRate = d.elec_rate || 7.0;

        document.getElementById('coord_display').innerHTML =
            '📍 '+d.location_name+' | Lat: '+d.map.lat+', Lon: '+d.map.lon;

        document.getElementById('status_box').innerHTML =
            '✅ Coordinates found | ☀️ PSH: '+d.map.annual_psh+'h/day | '+
            '⚡ Rate: ₹'+d.elec_rate+'/kWh ('+d.state_name+') | '+
            '🏪 Kenbrook prices scraped | 🧠 AI learned from '+d.feedback_count+' past predictions';

        if(d.map && d.map.lat){
            document.getElementById('map_section').style.display='block';
            document.getElementById('map_frame').src=
                'https://www.openstreetmap.org/export/embed.html?bbox='+
                (parseFloat(d.map.lon)-0.05)+','+(parseFloat(d.map.lat)-0.05)+','+
                (parseFloat(d.map.lon)+0.05)+','+(parseFloat(d.map.lat)+0.05)+
                '&layer=mapnik&marker='+d.map.lat+','+d.map.lon;

            document.getElementById('source_info').innerHTML=`
                <div class="source-box">
                    📊 All Sources Scraped:<br>
                    <span class="nasa-badge">🛰️ NASA: ${d.map.nasa_psh||'N/A'}h/day</span>
                    <span class="pvgis-badge">🌍 PVGIS: ${d.map.pvgis_psh||'N/A'}h/day</span>
                    <span class="sun-badge">☀️ Final PSH: ${d.map.annual_psh}h/day</span>
                    <span class="kenbrook-badge">🏪 Panel: ₹${d.prices.panel_price}</span>
                    <span class="elec-badge">⚡ Rate: ₹${d.elec_rate}/kWh (${d.state_name})</span>
                </div>
            `;

            document.getElementById('weather_box').innerHTML=`
                <div class="w-card"><div class="wval">${d.map.annual_psh}h</div><div class="wlbl">Final PSH</div></div>
                <div class="w-card"><div class="wval">${d.map.nasa_psh||'N/A'}h</div><div class="wlbl">NASA PSH</div></div>
                <div class="w-card"><div class="wval">${d.map.pvgis_psh||'N/A'}h</div><div class="wlbl">PVGIS PSH</div></div>
                <div class="w-card"><div class="wval">${d.map.effective_sun}h</div><div class="wlbl">Effective PSH</div></div>
                <div class="w-card"><div class="wval">${d.map.summer_psh}h</div><div class="wlbl">Summer PSH</div></div>
                <div class="w-card"><div class="wval">${d.map.monsoon_psh}h</div><div class="wlbl">Monsoon PSH</div></div>
                <div class="w-card"><div class="wval">${d.map.winter_psh}h</div><div class="wlbl">Winter PSH</div></div>
                <div class="w-card"><div class="wval">${d.map.temp}°C</div><div class="wlbl">Live Temp</div></div>
                <div class="w-card"><div class="wval">${d.map.cloud}%</div><div class="wlbl">Cloud Cover</div></div>
                <div class="w-card"><div class="wval">${d.map.uv}</div><div class="wlbl">UV Index</div></div>
                <div class="w-card"><div class="wval">₹${d.elec_rate}</div><div class="wlbl">Electricity Rate</div></div>
                <div class="w-card"><div class="wval">${d.map.annual_temp}°C</div><div class="wlbl">Annual Temp</div></div>
            `;
        }

        showResults(d);
        document.getElementById('ai_answer').innerHTML = d.report;

        lastPrediction = {
            lat: d.map.lat,
            lon: d.map.lon,
            location: d.location_name,
            predicted_psh: d.map.annual_psh,
            predicted_rate: d.elec_rate,
            state: d.state_name
        };
        document.getElementById('feedback_box').style.display='block';
        loadFeedbackStats();

    }catch(e){
        document.getElementById('ai_answer').innerHTML='Error: '+e.message;
    }
}

function showResults(d){
    const area = parseFloat(document.getElementById('area').value)||1000;
    const panelType = document.getElementById('paneltype').value;
    const eff = {"Monocrystalline":1.0,"Polycrystalline":0.85,"Thin Film":0.70}[panelType];
    const panels = Math.floor(area*0.7/1.7);
    const capacityKW = Math.round(panels*0.4);
    const dailyKWh = Math.round(capacityKW*annualPSH*eff);
    const yearlyKWh = dailyKWh*365;
    const yearlyIncome = Math.round(yearlyKWh*elecRate);
    const panelCost = panels*panelPrice;
    const inverterCost = capacityKW*inverterPrice;
    const cableDC = Math.round(area*0.8);
    const cableAC = Math.round(area*0.3);
    const cableCost = Math.round((cableDC+cableAC)*150);
    const mountCost = panels*3000;
    const battCost = capacityKW*5000;
    const installCost = capacityKW*10000;
    const miscCost = Math.round((panelCost+inverterCost+cableCost+mountCost+battCost+installCost)*0.1);
    const total = panelCost+inverterCost+cableCost+mountCost+battCost+installCost+miscCost;
    const years = (total/yearlyIncome).toFixed(1);

    document.getElementById('metrics').innerHTML=`
        <div class="metric"><div class="val">${panels}</div><div class="lbl">Solar Panels</div></div>
        <div class="metric"><div class="val">${capacityKW} kW</div><div class="lbl">Capacity</div></div>
        <div class="metric"><div class="val">${dailyKWh} kWh</div><div class="lbl">Daily Output</div></div>
        <div class="metric"><div class="val">${years} yrs</div><div class="lbl">ROI Recovery</div></div>
    `;
    document.getElementById('equipment').innerHTML=`
        <div class="item"><span class="item-name">Solar Panels (${panelType} 400W)</span><span class="item-val">${panels} units @ ₹${panelPrice.toLocaleString()}/unit</span></div>
        <div class="item"><span class="item-name">Grid-Tie Inverter</span><span class="item-val">${capacityKW} kW @ ₹${inverterPrice.toLocaleString()}/kW</span></div>
        <div class="item"><span class="item-name">DC Cables</span><span class="item-val">${cableDC} meters</span></div>
        <div class="item"><span class="item-name">AC Cables</span><span class="item-val">${cableAC} meters</span></div>
        <div class="item"><span class="item-name">Mounting Structure</span><span class="item-val">${panels} sets</span></div>
        <div class="item"><span class="item-name">Battery Bank</span><span class="item-val">${capacityKW*2} kWh</span></div>
        <div class="item"><span class="item-name">Junction Boxes</span><span class="item-val">${Math.ceil(panels/10)} units</span></div>
        <div class="item"><span class="item-name">Earthing System</span><span class="item-val">1 complete set</span></div>
        <div class="item"><span class="item-name">Net Energy Meter</span><span class="item-val">1 unit</span></div>
        <div class="item"><span class="item-name">CCTV Monitoring</span><span class="item-val">1 set</span></div>
    `;
    document.getElementById('costs').innerHTML=`
        <div class="item"><span class="item-name">Solar Panels (Kenbrook price)</span><span class="item-val">₹${(panelCost/100000).toFixed(2)}L</span></div>
        <div class="item"><span class="item-name">Inverter</span><span class="item-val">₹${(inverterCost/100000).toFixed(2)}L</span></div>
        <div class="item"><span class="item-name">Cables</span><span class="item-val">₹${(cableCost/100000).toFixed(2)}L</span></div>
        <div class="item"><span class="item-name">Mounting Structure</span><span class="item-val">₹${(mountCost/100000).toFixed(2)}L</span></div>
        <div class="item"><span class="item-name">Battery Bank</span><span class="item-val">₹${(battCost/100000).toFixed(2)}L</span></div>
        <div class="item"><span class="item-name">Installation</span><span class="item-val">₹${(installCost/100000).toFixed(2)}L</span></div>
        <div class="item"><span class="item-name">Misc (10%)</span><span class="item-val">₹${(miscCost/100000).toFixed(2)}L</span></div>
        <div class="total"><span>Total Investment</span><span>₹${(total/100000).toFixed(2)} Lakhs</span></div>
    `;
    document.getElementById('roi').innerHTML=`
        <div class="item"><span class="item-name">NASA+PVGIS PSH</span><span class="item-val">${annualPSH}h/day</span></div>
        <div class="item"><span class="item-name">Auto-Scraped Rate</span><span class="item-val">₹${elecRate}/kWh</span></div>
        <div class="item"><span class="item-name">Total Investment</span><span class="item-val">₹${(total/100000).toFixed(2)}L</span></div>
        <div class="item"><span class="item-name">Yearly Output</span><span class="item-val">${yearlyKWh.toLocaleString()} kWh</span></div>
        <div class="item"><span class="item-name">Yearly Income</span><span class="item-val">₹${(yearlyIncome/100000).toFixed(2)}L/year</span></div>
        <div class="item"><span class="item-name">Payback Period</span><span class="item-val"><span class="tag">${years} Years</span></span></div>
        <div class="item"><span class="item-name">25 Year Earnings</span><span class="item-val">₹${((yearlyIncome*25)/100000).toFixed(0)}L</span></div>
        <div class="item"><span class="item-name">CO2 Saved/Year</span><span class="item-val">${Math.round(yearlyKWh*0.82/1000)} Tonnes</span></div>
    `;
}

async function sendFeedback(score){
    const correction = document.getElementById('correction').value;
    try{
        await fetch('/feedback',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({score, correction, prediction: lastPrediction})
        });
        if(score === 1){
            alert('✅ Thanks! AI noted this was correct for '+lastPrediction.location);
        } else {
            alert('❌ Thanks! AI will learn from this mistake for '+lastPrediction.location);
        }
        document.getElementById('correction').value='';
        loadFeedbackStats();
    }catch(e){
        alert('Error saving feedback: '+e.message);
    }
}

async function loadFeedbackStats(){
    try{
        const r = await fetch('/feedback_stats');
        const d = await r.json();
        document.getElementById('feedback_stats').innerHTML =
            '🧠 AI has learned from '+d.total+' predictions | '+
            '👍 Correct: '+d.correct+' | 👎 Wrong: '+d.wrong+
            ' | Accuracy: '+d.accuracy+'%';
    }catch(e){}
}
</script>
</body>
</html>
"""

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []

def save_feedback(data):
    existing = load_feedback()
    existing.append(data)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(existing, f, indent=2)

def get_learned_correction(lat, lon):
    feedback = load_feedback()
    corrections = []
    for fb in reversed(feedback):
        if fb.get("score") == -1 and fb.get("correction"):
            try:
                prev_lat = float(fb["prediction"]["lat"])
                prev_lon = float(fb["prediction"]["lon"])
                if abs(prev_lat-float(lat))<1.0 and abs(prev_lon-float(lon))<1.0:
                    corrections.append(fb["correction"])
            except:
                pass
    if corrections:
        return "PAST USER CORRECTIONS FOR NEARBY LOCATIONS:\n" + "\n".join(f"- {c}" for c in corrections[-3:])
    return ""

def get_coordinates(lat, lon, loc_name):
    if lat and lon and str(lat).strip() and str(lon).strip():
        try:
            reverse_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
            headers = {"User-Agent": "SolarPlanner/1.0"}
            r = requests.get(reverse_url, headers=headers, timeout=10).json()
            name = r.get("display_name", f"Lat {lat}, Lon {lon}")[:60]
        except:
            name = f"Lat {lat}, Lon {lon}"
        return float(lat), float(lon), name

    print(f"Finding coordinates for: {loc_name}")
    geo_url = f"https://nominatim.openstreetmap.org/search?q={loc_name},India&format=json&limit=1"
    headers = {"User-Agent": "SolarPlanner/1.0"}
    geo = requests.get(geo_url, headers=headers, timeout=10).json()
    if geo:
        lat = round(float(geo[0]["lat"]), 4)
        lon = round(float(geo[0]["lon"]), 4)
        name = geo[0]["display_name"][:60]
        print(f"Found: {lat}, {lon}")
        return lat, lon, name
    raise Exception("Could not find coordinates!")

def scrape_electricity_rate(lat, lon):
    try:
        print("Finding state from coordinates...")
        geo_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "SolarPlanner/1.0"}
        geo = requests.get(geo_url, headers=headers, timeout=10).json()
        state = geo.get("address", {}).get("state", "").lower()
        print(f"State: {state}")

        state_rates = {
            "andhra pradesh": 6.50, "telangana": 7.00,
            "karnataka": 7.20, "tamil nadu": 6.80,
            "kerala": 6.40, "maharashtra": 8.50,
            "gujarat": 6.00, "rajasthan": 7.50,
            "madhya pradesh": 7.00, "uttar pradesh": 7.00,
            "delhi": 8.00, "haryana": 7.50,
            "punjab": 7.00, "west bengal": 7.50,
            "odisha": 6.00, "jharkhand": 6.50,
            "bihar": 7.00, "assam": 6.50,
            "himachal pradesh": 5.50, "uttarakhand": 6.00,
            "goa": 4.50, "chhattisgarh": 6.00,
            "tripura": 6.00, "meghalaya": 6.00,
        }

        try:
            print("Scraping live rate from bijlibachao.com...")
            url = "https://www.bijlibachao.com/electricity-tariff/electricity-tariff-rate-in-india.html"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            state_words = state.split()
            for word in state_words:
                if len(word) < 3:
                    continue
                pattern = rf'{word}[^.]*?(\d+\.?\d*)\s*(?:per\s*unit|per unit|₹/unit)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    live_rate = float(match.group(1))
                    if 2.0 < live_rate < 20.0:
                        print(f"Live rate: ₹{live_rate} for {state}")
                        return live_rate, state, "bijlibachao.com (live)"
        except Exception as e:
            print(f"Live scrape error: {e}")

        for key, rate in state_rates.items():
            if key in state or state in key:
                print(f"Table rate: ₹{rate} for {state}")
                return rate, state, "state tariff table"

        return 7.0, state or "unknown", "default estimate"

    except Exception as e:
        print(f"Rate error: {e}")
        return 7.0, "unknown", "default"

def scrape_kenbrook():
    try:
        print("Scraping Kenbrook Solar...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        panel_price = 18000
        inverter_price = 8000

        r = requests.get("https://kenbrooksolar.com/solar-panel-price-in-india", headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        prices = re.findall(r'(?:Rs\.?|₹)\s*([0-9,]+)', text)
        if prices:
            price_nums = [int(p.replace(',','')) for p in prices if 5000 < int(p.replace(',','')) < 50000]
            if price_nums:
                panel_price = int(sum(price_nums)/len(price_nums))

        try:
            r2 = requests.get("https://kenbrooksolar.com/solar-inverter-price", headers=headers, timeout=15)
            soup2 = BeautifulSoup(r2.text, "html.parser")
            text2 = soup2.get_text(separator=" ", strip=True)
            prices2 = re.findall(r'(?:Rs\.?|₹)\s*([0-9,]+)', text2)
            if prices2:
                inv_prices = [int(p.replace(',','')) for p in prices2 if 3000 < int(p.replace(',','')) < 100000]
                if inv_prices:
                    inverter_price = int(sum(inv_prices)/len(inv_prices))
        except Exception as e:
            print(f"Inverter scrape error: {e}")

        print(f"Kenbrook: panel=₹{panel_price}, inverter=₹{inverter_price}")
        return {"panel_price": panel_price, "inverter_price": inverter_price}
    except Exception as e:
        print(f"Kenbrook error: {e}")
        return {"panel_price": 18000, "inverter_price": 8000}

def get_solar_data(lat, lon):
    result = {}

    try:
        print("Scraping NASA POWER...")
        nasa_url = (
            f"https://power.larc.nasa.gov/api/temporal/climatology/point"
            f"?parameters=ALLSKY_SFC_SW_DWN,T2M,WS10M,CLOUD_AMT"
            f"&community=RE&longitude={lon}&latitude={lat}&format=JSON"
        )
        nasa = requests.get(nasa_url, timeout=30).json()
        solar = nasa['properties']['parameter']['ALLSKY_SFC_SW_DWN']
        temp_data = nasa['properties']['parameter']['T2M']
        wind_data = nasa['properties']['parameter']['WS10M']
        months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
        monthly_psh = {m: round(float(solar[m]),2) for m in months}
        nasa_psh = round(float(solar['ANN']),2)
        result['nasa_psh'] = nasa_psh
        result['monthly_psh'] = monthly_psh
        result['summer_psh'] = round((monthly_psh['MAR']+monthly_psh['APR']+monthly_psh['MAY'])/3,2)
        result['monsoon_psh'] = round((monthly_psh['JUN']+monthly_psh['JUL']+monthly_psh['AUG'])/3,2)
        result['winter_psh'] = round((monthly_psh['DEC']+monthly_psh['JAN']+monthly_psh['FEB'])/3,2)
        result['annual_temp'] = round(float(temp_data['ANN']),1)
        result['annual_wind'] = round(float(wind_data['ANN']),1)
        print(f"NASA PSH: {nasa_psh}")
    except Exception as e:
        print(f"NASA error: {e}")
        result['nasa_psh'] = None
        result['summer_psh'] = 0
        result['monsoon_psh'] = 0
        result['winter_psh'] = 0
        result['annual_temp'] = "N/A"
        result['annual_wind'] = "N/A"
        result['monthly_psh'] = {}

    try:
        print("Scraping PVGIS...")
        pvgis_url = (
            f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
            f"?lat={lat}&lon={lon}&peakpower=1&loss=14"
            f"&outputformat=json&browser=0"
        )
        pvgis = requests.get(pvgis_url, timeout=20).json()
        yearly_kwh = pvgis['outputs']['totals']['fixed']['E_y']
        pvgis_psh = round(yearly_kwh/365, 2)
        result['pvgis_psh'] = pvgis_psh
        print(f"PVGIS PSH: {pvgis_psh}")
    except Exception as e:
        print(f"PVGIS error: {e}")
        result['pvgis_psh'] = None

    try:
        print("Scraping live weather...")
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,cloud_cover,wind_speed_10m,uv_index"
            f"&timezone=auto"
        )
        live = requests.get(weather_url, timeout=10).json()
        current = live.get("current",{})
        result['temp'] = current.get("temperature_2m","N/A")
        result['cloud'] = current.get("cloud_cover",0)
        result['uv'] = current.get("uv_index","N/A")
        result['wind'] = current.get("wind_speed_10m","N/A")
    except Exception as e:
        print(f"Weather error: {e}")
        result['temp'] = "N/A"
        result['cloud'] = 0
        result['uv'] = "N/A"
        result['wind'] = "N/A"

    valid = [v for v in [result.get('nasa_psh'), result.get('pvgis_psh')] if v]
    if valid:
        final_psh = round(sum(valid)/len(valid), 2)
    else:
        lat_f = float(lat)
        if lat_f > 32: final_psh = 3.8
        elif lat_f > 28: final_psh = 4.5
        elif lat_f > 23: final_psh = 5.2
        elif lat_f > 18: final_psh = 5.5
        elif lat_f > 13: final_psh = 5.3
        else: final_psh = 5.0

    cloud_factor = 1-(float(result.get('cloud',0))/100*0.75)
    result['annual_psh'] = final_psh
    result['effective_sun'] = round(final_psh*cloud_factor, 2)
    result['lat'] = float(lat)
    result['lon'] = float(lon)
    print(f"FINAL PSH: {final_psh}")
    return result

def ask_ai(lat, lon, location_name, area, panel_type, elec_rate, projtype, weather, prices):
    client = Groq(api_key=GROQ_KEY)
    learned = get_learned_correction(lat, lon)
    annual_psh = weather.get("annual_psh", 5.0)
    effective_sun = weather.get("effective_sun", 4.5)
    cloud = weather.get("cloud", 0)
    panel_price = prices.get("panel_price", 18000)
    inverter_price = prices.get("inverter_price", 8000)
    eff = {"Monocrystalline":1.0,"Polycrystalline":0.85,"Thin Film":0.70}.get(panel_type, 1.0)
    panels = int(float(area)*0.7/1.7)
    capacity_kw = round(panels*0.4)
    daily_kwh = round(capacity_kw*effective_sun*eff)
    yearly_kwh = daily_kwh*365
    yearly_income = round(yearly_kwh*float(elec_rate))
    panel_cost = panels*panel_price
    inverter_cost = capacity_kw*inverter_price
    other_cost = (int(float(area))*150)+(panels*3000)+(capacity_kw*5000)+(capacity_kw*10000)
    total_cost = round((panel_cost+inverter_cost+other_cost)*1.1)
    years = round(total_cost/yearly_income, 1)

    prompt = f"""
You are a solar energy expert in India. Analyze this project:

{learned}

LOCATION: {location_name}
COORDINATES: Lat {lat}, Lon {lon}
PROJECT: {projtype} | Area: {area} sqm | {panel_type}
PANELS: {panels} | Capacity: {capacity_kw} kW

SCRAPED DATA:
- NASA PSH: {weather.get('nasa_psh','N/A')}h/day
- PVGIS PSH: {weather.get('pvgis_psh','N/A')}h/day
- Final PSH: {annual_psh}h/day
- Summer: {weather.get('summer_psh')}h | Monsoon: {weather.get('monsoon_psh')}h | Winter: {weather.get('winter_psh')}h
- Temp: {weather.get('annual_temp')}C | Cloud: {cloud}%
- Kenbrook Panel: Rs {panel_price}/unit
- Auto-Scraped Electricity Rate: Rs {elec_rate}/kWh

RESULTS:
- Daily Output: {daily_kwh} kWh
- Yearly Income: Rs {yearly_income:,}
- Investment: Rs {total_cost:,}
- ROI: {years} years

Provide:
1. Location suitability at {location_name}
2. Why {annual_psh} PSH at this location
3. Seasonal income variation
4. Suitability score out of 10
5. PM Surya Ghar Yojana subsidy
6. Best panel tilt for Lat {lat}
7. Installation timeline
8. Final verdict — ROI in {years} years worth it?
"""
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        max_tokens=1500
    )
    header = f"📍 {location_name}\n🌐 {lat}, {lon} | ☀️ PSH: {annual_psh}h | ⚡ ₹{elec_rate}/kWh | 📈 ROI: {years} yrs\n{'='*50}\n\n"
    return header + r.choices[0].message.content

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        lat, lon, location_name = get_coordinates(
            data.get("lat","").strip(),
            data.get("lon","").strip(),
            data.get("locName","")
        )
        elec_rate, state_name, rate_source = scrape_electricity_rate(lat, lon)
        weather = get_solar_data(lat, lon)
        prices = scrape_kenbrook()
        feedback_data = load_feedback()
        report = ask_ai(lat,lon,location_name,data["area"],data["panelType"],elec_rate,data["projtype"],weather,prices)
        return jsonify({
            "report": report,
            "map": weather,
            "prices": prices,
            "location_name": location_name,
            "elec_rate": elec_rate,
            "state_name": state_name,
            "rate_source": rate_source,
            "feedback_count": len(feedback_data)
        })
    except Exception as e:
        return jsonify({"error": str(e), "map":{}, "prices":{}})

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.json
        data["timestamp"] = datetime.now().isoformat()
        save_feedback(data)
        print(f"Feedback: score={data['score']}, correction={data.get('correction','')}")
        return jsonify({"status": "saved"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/feedback_stats")
def feedback_stats():
    try:
        data = load_feedback()
        correct = sum(1 for f in data if f.get("score") == 1)
        wrong = sum(1 for f in data if f.get("score") == -1)
        total = len(data)
        accuracy = round(correct/total*100) if total > 0 else 0
        return jsonify({"total":total,"correct":correct,"wrong":wrong,"accuracy":accuracy})
    except Exception as e:
        return jsonify({"total":0,"correct":0,"wrong":0,"accuracy":0})

if __name__ == "__main__":
    app.run(debug=False, threaded=True)


#  Natural Language–Driven Agentic Web Information Retrieval System
## Solar Plant AI Planner

**[Project Demo on YouTube](#)** *(https://youtu.be/WIfwOwRmu_w?si=Lg3IVOiFYgo4i-iA)*

---

##  Project Overview

The **Solar Plant AI Planner** is an intelligent, multimodal web scraping agent that takes land coordinates or location name as input and automatically calculates solar plant requirements, real-time costs, ROI analysis, and feasibility — powered by LLM-based reasoning and multi-source data scraping.

The system combines **Natural Language Processing**, **Agentic Web Scraping**, and **Multimodal AI** to deliver actionable solar investment reports for any location in India.

---

##  Team Members

| Name | Role | Roll No. |
|------|------|----|
| Vedant | Team Lead | 524188 |
| Shriya Rathaur | Team Member | 524175 |
| Sai Charan | Team Member  | 524158 |

---

##  Key Features

###  Multi-Source Real-Time Scraping
Scrapes live data from multiple sources simultaneously:
- **NASA POWER API** — Historical solar radiation & peak sun hours
- **PVGIS (EU Commission)** — Solar energy yield per location
- **Open-Meteo** — Live weather, temperature, cloud cover, UV index
- **Kenbrook Solar** — Real panel and inverter market prices
- **Bijli Bachao** — State-wise electricity tariff rates
- **OpenStreetMap** — Coordinate geocoding and reverse geocoding

###  Intelligent AI Analysis
- LLaMA 3.3 70B via Groq API for expert solar analysis
- Auto-detects location from name or coordinates
- Seasonal variation analysis (Summer / Monsoon / Winter PSH)
- PM Surya Ghar Yojana subsidy recommendations
- Optimal panel tilt angle calculation

###  Smart Location Detection
- Enter **coordinates** → system maps location and finds state
- Enter **location name** → system auto-finds exact lat/lon
- Interactive OpenStreetMap showing your exact land

###  Feedback-Based Learning (RLHF)
- Users rate predictions as correct or wrong
- System stores corrections for nearby locations
- AI improves future predictions using past feedback
- Tracks accuracy statistics over time

###  User-Friendly Web Interface
- Flask-based real-time web UI
- No manual electricity rate input needed — auto-scraped by state
- Live weather cards and solar data badges
- Equipment list with live Kenbrook prices

---

##  Technical Architecture

### Core Components

**Web Scraping Layer**
- BeautifulSoup4 + Requests for HTML parsing
- Multi-source parallel scraping (NASA, PVGIS, Kenbrook, Bijli Bachao)
- Geocoding via Nominatim OpenStreetMap API

**Solar Calculation Engine**
- Peak Sun Hours averaged from NASA + PVGIS for accuracy
- Cloud cover correction factor applied to effective PSH
- Panel count formula: `(Area × 0.70) ÷ 1.7 sqm`
- ROI calculation using auto-scraped electricity rates

**LLM Integration**
- Groq LLaMA 3.3 70B for natural language analysis
- Location-specific recommendations
- Past feedback injected into prompts for learning

**Feedback System**
- JSON-based feedback storage
- Nearby location matching (within 1° radius)
- Accuracy tracking dashboard

---

##  Technology Stack

| Layer | Tools Used |
|-------|-----------|
| Backend | Python 3.10+, Flask, Requests, BeautifulSoup4 |
| AI / LLM | Groq API, LLaMA 3.3 70B (free tier) |
| Solar Data | NASA POWER API, PVGIS EU API |
| Weather | Open-Meteo API |
| Geocoding | OpenStreetMap Nominatim |
| Price Scraping | Kenbrook Solar, Bijli Bachao |
| Maps | OpenStreetMap Embed |
| Feedback | JSON-based RLHF storage |

---

##  Setup Instructions

###  Prerequisites
- Python 3.10+
- Internet connection
- Free Groq API key
- Git

###  Installation

```bash
git clone [your-repository-url]
cd WebScrapingAgent
pip install flask groq requests beautifulsoup4
```

###  API Key Configuration

Open `solar.py` and replace line 7:

```python
GROQ_KEY = "paste_your_gsk_key_here"
```

Get your free key at [console.groq.com](https://console.groq.com) → API Keys → Create API Key

### ▶️ Running the Application

```bash
python solar.py
```

Then visit: `http://127.0.0.1:5000`

---

##  Usage Guide

###  Enter Location
Enter coordinates directly OR just type a location name:
```
Option 1: Latitude 17.9784, Longitude 79.5300
Option 2: Type "NIT Warangal, Telangana"
```

###  Configure Project
- Area in square meters (e.g. 1000)
- Panel type: Monocrystalline / Polycrystalline / Thin Film
- Project type: Residential / Commercial / Industrial / Agricultural

###  Get AI Report
Click **Scrape All Sources + AI Report + Map** and wait 30-60 seconds.

The system will:
1. Auto-find coordinates from location name
2. Scrape NASA POWER + PVGIS solar data
3. Fetch live weather at those coordinates
4. Scrape Kenbrook Solar panel prices
5. Auto-detect state electricity rate
6. Generate full AI analysis with ROI

###  Feedback and Learning
Rate the prediction accuracy using thumbs up / thumbs down.
Add corrections like "Actual PSH was 4.2h not 5.0h".
The AI uses this to improve future nearby predictions.

---

##  Calculation Formula

```
Panels Required   = (Area × 0.70) ÷ 1.7
Capacity (kW)     = Panels × 0.4
Daily Output      = Capacity × Effective PSH × Panel Efficiency
Yearly Output     = Daily Output × 365
Yearly Income     = Yearly Output × Electricity Rate (auto-scraped)
Total Investment  = (Panels + Inverter + Cables + Mount + Battery + Install) × 1.10
ROI Years         = Total Investment ÷ Yearly Income
```

**Panel Efficiency:** Monocrystalline 100% | Polycrystalline 85% | Thin Film 70%

---

##  Project Structure

```
WebScrapingAgent/
├── solar.py              # Main Flask application (Solar AI Planner)
├── app.py                # General web scraping agent
├── agent.py              # Core scraping + BeautifulSoup module
├── feedback_data.json    # RLHF feedback storage (auto-created)
├── results_*.txt         # Auto-saved scraping results
└── README.md             # This file
```

---

##  Sample Output

```
Location   : NIT Warangal, Telangana
Coordinates: 17.9784, 79.5300
NASA PSH   : 5.47h/day | PVGIS: 5.31h/day | Final: 5.39h/day
Elec Rate  : Rs 7.00/kWh (Telangana) — auto scraped
Panel Price: Rs 18,500/unit (Kenbrook Solar)

Panels Required : 411 units
Total Capacity  : 164 kW
Daily Output    : 885 kWh/day
Yearly Income   : Rs 22.6 Lakhs/year
Total Investment: Rs 1.85 Crore
ROI Recovery    : 8.2 years
25 Year Earnings: Rs 5.65 Crore
CO2 Saved/Year  : 295 Tonnes
```

---

##  Future Enhancements

-  Database storage for prediction history
-  Real-time panel price API integration
-  Multi-language support (Telugu, Hindi)
-  Cloud deployment (AWS / GCP)
-  Mobile-responsive UI
-  PDF report generation
-  Battery sizing optimization module

---

##  Contributing

Open to feature suggestions, bug reports, or pull requests. Let's build together!

---

##  Notes

- Prices are estimates — always get 3 quotes before purchasing
- NASA POWER API takes 15-30 seconds to respond
- Groq free tier has rate limits — avoid rapid repeated requests


---

*Built with Python · Flask · Groq LLaMA 3.3 · NASA POWER · PVGIS · BeautifulSoup4 · OpenStreetMap*

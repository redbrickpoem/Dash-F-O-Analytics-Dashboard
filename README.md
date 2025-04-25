# 📈 F&O Options Analytics Dashboard

This is a fully-featured options analytics dashboard built using Python and Dash. It allows users to visualize and analyze live F&O (Futures & Options) data, including Open Interest, IV, Volume, Option Greeks, Max Pain, and PCR metrics with interactive charts and filters.

## 🚀 Features

- 📊 Visualize OI, IV, Volume by Strike Price
- 🔁 Real-time data updates using NSE scraping
- ⚙️ Filter by Option Type, Expiry Date, and Strike Range
- 💡 Option Greeks (Delta, Gamma, Theta, Vega)
- 📉 Max Pain Point & PCR (Put-Call Ratio) calculation
- 📥 Export data as CSV
- 🧪 Clean, modular Dash layout using Bootstrap Components

## 🛠️ Tech Stack

- Python 3.x
- Dash (Plotly)
- Dash Bootstrap Components
- Pandas
- BeautifulSoup (NSE scraping)
- Gunicorn (for deployment)

## 📦 Installation

```bash
git clone https://github.com/your-username/fno-options-dashboard.git
cd fno-options-dashboard
pip install -r requirements.txt
python app.py

🌐 Deployment (Render)
This project is ready to be deployed on Render with a Procfile and requirements.txt.
Just connect the repo and Render will auto-detect it as a web service. (See full steps below.)

📁 Project Structure

├── app.py
├── components/
├── data/
├── utils/
├── assets/
├── requirements.txt
├── Procfile
├── README.md
└── .gitignore

📄 License
MIT License

🙌 Contribute
Feel free to fork, raise issues, or contribute to the project!

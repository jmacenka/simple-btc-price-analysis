# Simple BTC price analysis

This is a very simple app which was Vibe-Coded by ChatGPT-o1 in Feb 2025.

It pulls historic btc price charts and gives possibility to change the axis from lin-lin to lin-log or log-log.

It also evaluates the RSI (Relative Strength Indicator), calculates two rolling averages one more short-term with 6 month and one more long-term with 2 years. Also it assumes a power-law growth rate and builds a best-fit curve for the btc price.

Then it tries to find good times for buying or selling btc assuming that buying is more important than selling. It only uses lagging-indicators and does not "try to predict" and future developments other than extrapolating the assumed power-law correlation for the next couple of quarters.

Feel free to poke around in it.

Assuming that you have python3 installed and are on either a Linux or Mac system do the following to launch the app (locally!):

```bash
python3 -m pip install -r requirements.txt
```

To install the requirements. Best to use a virtual environment like [pipenv]() or [venv](), do as you see fit.

Then to launch the app locally run:

```bash
python3 app.py
```

and visit:

https://localhost:8050 to view the analysis which pulls the latest data from [yfinance]() for btc and calculates everything else locally.

Always do your own research! Nothing of this is financial advise. Stay safe and have fun!

> 19 Apr 2025, Jan
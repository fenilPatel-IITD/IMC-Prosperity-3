# 🧠 IMC Prosperity 3 – My Algo Trading Journey 📈

## 🌍 Global Rank: **36**  
## 🇮🇳 India Rank: **7**

---

## 🏁 Introduction

**IMC Prosperity 3** is a global algorithmic trading simulation hosted by **IMC Trading**, designed to give participants a hands-on experience of the real-world challenges faced by quantitative traders and developers.

Spanning over **15 days**, this high-octane competition is divided into **5 intense rounds**, each lasting **3 days**, testing our skills in:

- ✅ Algorithmic trading
- ✅ Chart and order book analysis
- ✅ Trend detection
- ✅ Dynamic pricing and arbitrage
- ✅ Options pricing and Greeks
- ✅ Manual trading and game theory

Each round brought a new level of complexity with evolving market dynamics and new products. We had to constantly adapt, experiment, and optimize strategies to stay competitive.

---

## 🌀 Round Breakdown

---

### 🔹 Round 1: Market Making & Fair Value Estimation

#### 🧠 Algo Part
- **Products**: AMETHYSTS, STARFRUIT
- **Objective**: Estimate fair value and make profitable markets
- **Strategy**:
  - Used mid-point of bid-ask as initial fair value
  - Placed symmetric buy/sell quotes around the fair price
  - Adjusted spread dynamically based on volatility and position

#### 🧠 Manual Part
The manual process for round 1 was fairly simple; we exhaustively tested all possible conversions and chose the most profitable option. This was the least time consuming manual round among all. (Similar to last years Round 2 manual challenge)

---

### 🔹 Round 2: Basket Trading & Derived Products

#### 🧠 Algo Part
- **New Mechanics**: Synthetic products (derived from baskets)
- **Strategy**:
  - Calculated basket fair value from component prices
  - Arbitraged price differences between baskets and constituents
  - Implemented safeguards against overexposure

#### 🧠 Manual Part
This round stood out as one of the most engaging ones, as it was a direct application of **Game Theory** with a psychological twist. We were asked to pick numbers that were **either multiples of 10 or prime numbers**—a seemingly simple task, but loaded with strategic nuance. I suspected that **73** would be a popular choice, and that hunch was spot-on—it turned out to be the most commonly selected number, as also discussed in this fascinating [Veritasium video](https://youtu.be/d6iQrh2TK98?si=sxUMLYYUGet8a5QO).

After carefully analyzing the risk and reward, we opted for a **safer choice: 90**. Choosing a **second container** was never seriously considered due to the high risk—due to less number of containers to choose from (10). We only found out **after the round ended** that the **maximum gain** from selecting a second container was just **5,000 seashells more than the 50,000 fee**, which confirmed that our conservative approach was the right decision. (Similar to last years Round 3 manual challenge).

We also tried simulating with using some bots to behave like humans, but it was not so successful.

---

### 🔹 Round 3: Trend Detection & Dynamic Strategies

#### 🧠 Algo Part
- **Market Behavior**: Strongly trending assets
- **Strategy**:
  - Used moving averages and momentum indicators
  - Added stop-loss and trailing profit logic
  - Optional: Used Point & Figure based trend logic for signals

#### 🧠 Manual Part
The first bid for the flippers was pretty straightforward—**200 seashells per flipper** was the obvious baseline. But things got interesting with the **second bid**, which we estimated would hover around **285**. To back it up, we even looked into **last year’s Round 4 manual results**, where the average winning bid was typically **2 seashells higher than the optimal answer**.

However, fueled by my recent dive into **behavioral psychology**, I suspected the average might skew toward a **round number like 290**—a common cognitive bias. So, taking a slightly aggressive stance, we went with **292**.

Unfortunately, the crowd played it cooler this time—the actual average ended up being just **1 seashell above the optimal bid**. Close, but not quite. 🤦🏻‍♂️

---

### 🔹 Round 4: Options Trading – Greeks and Volatility

#### 🧠 Algo Part
- **Products**: Underlying + multiple strike options
- **Strategy**:
  - Implemented Black-Scholes pricing
  - Estimated implied volatility from market quotes
  - Hedged using delta/gamma neutral portfolios
  - Identified arbitrage via vertical and butterfly spreads

#### 🧠 Manual Part
This manual challenge was reminiscent of **Round 2** and **last year’s Round 4**, but with a twist—the moderators **provided the actual data from Round 2**, which made things even more interesting... and unpredictable. With that information out in the open, it became significantly harder to **model participant behavior**, as everyone had access to similar reference points.

Adding to the challenge, our team was juggling **academic commitments** during this period, which limited the time we could dedicate to analyzing patterns or devising a robust strategy. As a result, we weren’t able to go as deep into this round as we had hoped.

---

### 🔹 Round 5: The Grand Finale – All-In-One Market Chaos

#### 🧠 Algo Part
- **All Concepts Combined**: Market making, baskets, trends, options
- **Strategy**:
  - Merged multiple strategies under a hybrid engine
  - Switched modes between arbitrage, market making, and trend following
  - Managed overall portfolio risk, positions, and liquidation timing

#### 🧠 Manual Part
This round closely mirrored **last year’s Round 5 manual challenge**, so we decided not to spend too much time overanalyzing it. Instead, we took a direct approach—**mapping the severity of this year’s news to last year’s**, assuming a near **1-to-1 correlation** between the items.

For the most part, this strategy held up well. However, two key exceptions—**Red-Flag** and **Solar-Panels**—behaved differently, which led to **lower profits than we had anticipated**.

By this point in the competition, we were also **running on minimal sleep**, which definitely affected our focus and decision-making. 😴

---

## 🚀 Final Thoughts

This competition pushed me to my limits—technically, analytically, and mentally. From building trading bots that adapted to market dynamics to solving game theory scenarios manually, **Prosperity 3 was a crash course in real-world trading**.

> **It wasn’t just about coding—it was about strategy, adaptability, and thinking like a trader.**

---

### 📌 What to Expect in This Repo:
- ✅ Code implementations for each round
- ✅ Write-ups and strategy docs
- ✅ Trading visualizations and analysis
- ✅ Key insights and learnings

---

Feel free to ⭐ the repo if you find it helpful or interesting!

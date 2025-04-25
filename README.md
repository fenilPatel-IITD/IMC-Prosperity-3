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
In the opening round, we were introduced to three products: RAINFOREST_RESIN, KELP, and SQUID_INK. RAINFOREST_RESIN was relatively straightforward—it behaved much like AMETHYSTS from the previous year, making it an ideal candidate for a basic symmetric market-making strategy centered around the mid-price. KELP, on the other hand, displayed more volatility, so we adopted a volume-weighted fair value approach to better capture short-term price movements. SQUID_INK proved to be the most challenging of the trio; its erratic behavior and lack of clear structure made it difficult to model effectively, and despite experimenting with various techniques, we were ultimately unable to formulate a profitable strategy for it during this round.

#### 🧠 Manual Part
The manual process for round 1 was fairly simple; we exhaustively tested all possible conversions and chose the most profitable option. This was the least time consuming manual round among all. (Similar to last years Round 2 manual challenge)

---

### 🔹 Round 2: Basket Trading & Derived Products

#### 🧠 Algo Part
Round 2 brought in a new wave of complexity with five additional products—PICNIC_BASKET1, PICNIC_BASKET2, CROISSANTS, DJEMBES, and JAMS. The structure of the baskets closely mirrored last year’s gift baskets, making it relatively straightforward to identify a profitable spread arbitrage opportunity. By comparing the combined fair value of the basket components against the actual basket price, we were able to consistently exploit mispricings, resulting in strong and stable returns during backtesting. The individual components were highly volatile, so instead of treating them as standalone assets, we primarily used their prices to inform our basket trades. Additionally, this round marked the point where we finally started trading SQUID_INK. After more experimentation, we deployed a basic market-making strategy around its mid-price to cautiously engage with its unpredictable behavior.

#### 🧠 Manual Part
This round stood out as one of the most engaging ones, as it was a direct application of **Game Theory** with a psychological twist. We were asked to pick numbers that were **either multiples of 10 or prime numbers**—a seemingly simple task, but loaded with strategic nuance. I suspected that **73** would be a popular choice, and that hunch was spot-on—it turned out to be the most commonly selected number, as also discussed in this fascinating [Veritasium video](https://youtu.be/d6iQrh2TK98?si=sxUMLYYUGet8a5QO).

After carefully analyzing the risk and reward, we opted for a **safer choice: 90**. Choosing a **second container** was never seriously considered due to the high risk—due to less number of containers to choose from (10). We only found out **after the round ended** that the **maximum gain** from selecting a second container was just **5,000 seashells more than the 50,000 fee**, which confirmed that our conservative approach was the right decision. (Similar to last years Round 3 manual challenge).

We also tried simulating with using some bots to behave like humans, but it was not so successful.

---

### 🔹 Round 3: Trend Detection & Dynamic Strategies

#### 🧠 Algo Part
In Round 3, the challenge level jumped significantly with the introduction of VOLCANIC_ROCK and five corresponding options—each with a different strike price. This was our first real encounter with options trading, and we were initially out of our depth. After some research, we began implementing the Black-Scholes Model to price the options, though early results were underwhelming. On the other hand, a simple trend-following strategy on the underlying VOLCANIC_ROCK asset yielded excellent performance in the backtests, so we leaned into that while continuing to explore options pricing. Midway through the round, the moderators revealed that the vouchers exhibited a volatility smile—deviating from the standard linear volatility used in traditional European options. Unfortunately, we didn’t have enough time to rework our model to incorporate this nuance, so we ultimately extended our trend-following logic to the vouchers as well, trading them directionally rather than by attempting precise theoretical pricing.

#### 🧠 Manual Part
The first bid for the flippers was pretty straightforward—**200 seashells per flipper** was the obvious baseline. But things got interesting with the **second bid**, which we estimated would hover around **285**. To back it up, we even looked into **last year’s Round 4 manual results**, where the average winning bid was typically **2 seashells higher than the optimal answer**.

However, fueled by my recent dive into **behavioral psychology**, I suspected the average might skew toward a **round number like 290**—a common cognitive bias. So, taking a slightly aggressive stance, we went with **292**.

Unfortunately, the crowd played it cooler this time—the actual average ended up being just **1 seashell above the optimal bid**. Close, but not quite. 🤦🏻‍♂️

---

### 🔹 Round 4: Options Trading – Greeks and Volatility

#### 🧠 Algo Part
Round 4 introduced MAGNIFICENT_MACARONS, a new product that could be traded within our archipelago or through conversions with Pristine Cuisine chefs, involving tariffs and transport fees. Initially, we experimented with conversions, but they didn’t provide the kind of profitability we expected. As a result, we focused on trading MAGNIFICENT_MACARONS solely within our archipelago, which proved to be more effective. We also explored correlations between MACARONS and the environmental factors mentioned in the round's observations, and found that sunlight played a key role in price movements, offering a promising strategy.

Meanwhile, Arpit worked on refining our approach to delta hedging for the vouchers, drawing on hints from the moderators in Discord. His work stabilized the results for both VOLCANIC_ROCK and the vouchers, especially when applied to the 10000 voucher. We also used linear regression to predict implied volatility (IV) for the 10000 voucher, which was then incorporated into a Black-Scholes model for pricing.

Toward the end of the round, the moderators provided a crucial hint about MACARONS: its price behavior changes depending on whether the sunlight index is above or below a specific threshold. This insight allowed us to refine our strategy, significantly improving performance and giving us a valuable edge as we concluded the round.

#### 🧠 Manual Part
This manual challenge was reminiscent of **Round 2** and **last year’s Round 4**, but with a twist—the moderators **provided the actual data from Round 2**, which made things even more interesting... and unpredictable. With that information out in the open, it became significantly harder to **model participant behavior**, as everyone had access to similar reference points.

Adding to the challenge, our team was juggling **academic commitments** during this period, which limited the time we could dedicate to analyzing patterns or devising a robust strategy. As a result, we weren’t able to go as deep into this round as we had hoped.

---

### 🔹 Round 5: The Grand Finale – All-In-One Market Chaos

#### 🧠 Algo Part
In Round 5, we went full throttle, building on our strong performance from the previous round. This round provided trade data from various bots, which we used to gain insights into their behaviors. To analyze this data, we created a Streamlit dashboard that allowed us to easily generate charts and identify patterns in the bot strategies across different products.

By quantitatively assigning scores to trade pairs, we uncovered valuable insights, particularly for CROISSANTS, which had previously been unprofitable for us. With this data, we improved our trading approach and turned CROISSANTS into a profitable product.

We also enhanced our basket strategy by fine-tuning parameters based on the new data, making it more profitable. For VOLCANIC_ROCK and vouchers, we continued applying delta hedging, avoiding trading out-of-the-money vouchers like 10,500 due to their lack of profitability in the last round.

The 10,000 voucher, which had earned us 120K in the previous round, remained a focus. We expected similar returns by applying the same strategy, with some adjustments based on the new insights.


#### 🧠 Manual Part
This round closely mirrored **last year’s Round 5 manual challenge**, so we decided not to spend too much time overanalyzing it. Instead, we took a direct approach—**mapping the severity of this year’s news to last year’s**, assuming a near **1-to-1 correlation** between the items.

For the most part, this strategy held up well. However, two key exceptions—**Red-Flag** and **Solar-Panels**—behaved differently, which led to **lower profits than we had anticipated**.

By this point in the competition, we were also **running on minimal sleep**, which definitely affected our focus and decision-making. 😴

---

## 🚀 Final Thoughts

IMC Prosperity 3 was undoubtedly the best competition we've ever participated in. The website, the organization, and the overall setup were top-notch. The challenges presented in each round were well-thought-out, pushing us to continuously adapt and innovate. Each new round introduced fresh obstacles, making the experience both exciting and rewarding. The team behind the competition did an excellent job creating a dynamic and immersive environment that truly tested our skills as algorithmic traders.

**It was a memorable journey, and we’re grateful for the opportunity to participate!**



---

### 📌 What to Expect in This Repo:
- ✅ Code implementations for each round
- ✅ Write-ups and strategy docs
- ✅ Trading visualizations and analysis
- ✅ Key insights and learnings

---

Feel free to ⭐ the repo if you find it helpful or interesting!

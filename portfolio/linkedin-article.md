# LinkedIn Article — Deep Portfolios with Autoencoders

---

## A. Headline Options

1. **Tracking 500 Stocks With 25: How a Neural Network Built My Index Portfolio (and Why My First Model Failed)**
2. **Deep Learning vs. PCA in Index Replication: Lessons From a Master's Project in AI & Quantum Finance**
3. **My Neural Network Reconstructed Better and Invested Worse: What Autoencoders Taught Me About Financial AI**

---

## B. Article Content (ready to paste)

# Tracking 500 Stocks With Just 25: What Building a Deep-Learning Index Tracker Taught Me

**Cover Image Suggestion:** A dark, minimal visual of a dense cloud of ~500 small dots (the index) flowing through a narrow neural-network "bottleneck" and coming out as 25 highlighted, color-coded nodes grouped by sector. Subtle stock-chart lines in the background; no stock-photo clichés.

---

Quantum computing and AI are often sold to finance as magic. In practice the hard part is less glamorous: getting a model to generalise on noisy, scarce market data, and proving that it beats a boring baseline.

That was the core lesson of a project I completed for my **Master's in AI & Quantum Computing Applied to Financial Markets (MIAX)**. The challenge sounded simple: replicate the **S&P 500 Equal Weight index** (about 500 stocks) with a portfolio of only **25 stocks**. The twist was that I wasn't allowed to pick the stocks. **A neural network I designed had to pick them.**

Here's what worked, what failed, and what I'd tell anyone applying deep learning to markets.

## The Financial Challenge

Index funds are simple in theory: buy everything in the index. In practice, holding 500 positions means transaction costs, operational complexity and rebalancing friction. A **sparse tracker**, a small portfolio that behaves like the whole index, solves that, as long as it tracks closely.

"Closely" is measured by **tracking error**: the annualised volatility of the daily return difference between your portfolio and the index. Lower is better.

The competition rules made this realistic:

- **Training:** 2015–2021 daily returns for 342 stocks.
- **Validation:** 2022–2023, a different regime, with the 2022 rate-hike bear market.
- **Test:** a hidden, random one-year window in 2024–2025, evaluated **once, irreversibly**, like putting a model into production.
- **Benchmarks to beat:** a classical **PCA** model (the "ghost competitor") and thousands of sector-balanced random portfolios ("monkeys").

There is also an irreducible floor. Even holding all 342 available stocks tracks the real ETF with about **2.2%** error, due to universe composition, survivorship bias and ETF mechanics. The game is getting as close to that floor as possible with 25 stocks.

## Bridging the Gap with Quantum & AI

The Master's program sits where **AI and quantum computing meet finance**. Much of the quantum content (optimisation, sampling, high-dimensional state spaces) comes down to one question: how do you find structure in a space too large to search naively? This project attacked that question with a **classical deep-learning** tool, the autoencoder.

The pipeline:

1. **Represent each stock as a sample.** Each of the 342 stocks becomes one row of ~1,762 daily returns (z-scored per day, fitted on training data only to avoid look-ahead bias).
2. **Compress with an autoencoder.** The network learns to reconstruct each stock's return history through a narrow bottleneck. That bottleneck becomes an **embedding**: a compact "fingerprint" of how the stock co-moves with the market.
3. **Group by direction, not magnitude.** Embeddings are projected onto the unit sphere and clustered into 25 groups with **spherical k-means** (cosine similarity). Two stocks are similar if they move *the same way*, not by the same amount.
4. **Pick real representatives.** Each cluster contributes its **medoid**, the most central *real* stock (not an abstract average), weighted by cluster size.
5. **Backtest honestly.** Quarterly rebalancing, exactly like the real ETF.

The selection and scoring steps were fixed and identical for every team. **The only lever was the quality of the embedding.** It was a pure test of representation learning.

## Overcoming Technical Hurdles

**The problem.** My first model was a single ReLU layer with a **400-dimensional** latent space, about **1.4 million parameters**. It trained smoothly, the loss curves looked healthy and the reconstruction error was *lower* than the final model's.

It still **lost to PCA**. My validation tracking error was 5.31%, worse than the linear baseline, and roughly 80% of the random "monkey" portfolios did better.

**The agitation.** This was the uncomfortable part. Everything I had been trained to look at said the model was good. But with only **342 samples and 1,762 features**, a large latent space doesn't compress anything. It memorises. The network had learned to copy stocks rather than to understand them. **The model that reconstructed best was the one that invested worst.** Adding more capacity, which is the intuitive fix, would have made it worse.

**The solution.** I redesigned the model around one principle: *force the network to find common factors, and give it no room to memorise.* Guided by the course's reference architecture, the final model is a **denoising autoencoder**:

- **Gaussian noise on the inputs** during training, so memorising daily noise stops paying off.
- **One small `tanh` hidden layer** (64 units) and an **8-dimensional linear bottleneck**. The linear output avoids "dead" units that break cosine-based clustering.
- **6x fewer parameters** (228k).

Result: **4.63% validation tracking error vs. 4.84% for PCA.** The model beat the ghost competitor in a bear market it had never seen.

**The engineering hurdles nobody mentions.** Turning the notebook into something someone else can trust took as much work as the model:

- Refactored it into a **modular, tested Python package** with 62 automated tests, centralised configuration and PEP 8 compliance.
- Enforced **full determinism**: two runs now produce byte-identical portfolios.
- Moved an API credential that had been hard-coded in the notebook into environment variables.
- Hit real-world friction along the way: TensorFlow not yet supporting the newest Python, and a Windows path-length limit breaking native libraries.

**The part I enjoyed most: explainability.** The network never saw sector labels, yet when I coloured the embedding by sector, **financials, healthcare, utilities, energy and tech formed distinct clusters.** One latent dimension correlated **0.74 with market beta** and 0.64 with volatility. The "black box" had rediscovered textbook finance on its own, which makes the model far easier to defend to a risk committee.

## Conclusion & Future Outlook

Three lessons I'm taking forward:

1. **Optimise the business metric, not the training loss.** Reconstruction error and tracking error disagreed, and only one of them pays.
2. **In finance, less capacity often means more generalisation.** Small data plus noisy signals favour simple, regularised models over deep ones.
3. **Validation is a tool, not a target.** With a one-shot test, resubmitting until the leaderboard looks good is just overfitting with extra steps.

Next, I'd like to explore **quantum and quantum-inspired approaches** for the portfolio-selection step. Choosing 25 stocks out of 342 is a combinatorial problem that maps naturally onto QUBO formulations and quantum annealing. I'd also compare autoencoder embeddings with variational and contrastive alternatives.

**Where have you seen a model "look good" on its training metric but fail on the metric that actually mattered?** I'd love to hear your war stories in the comments.

🔗 Full code, architecture diagrams and results: **[GitHub link here]**

---

## C. Strategic Hashtags

#MachineLearning #QuantitativeFinance #DeepLearning #FinTech #PortfolioManagement

---

## D. LinkedIn Feed Promotion Post

My neural network reconstructed the market better than PCA.
And invested worse. 📉

For my Master's in AI & Quantum Finance, I built a 25-stock portfolio that tracks the ~500-stock S&P 500 Equal Weight, with every stock chosen by an autoencoder.
The fix wasn't a bigger model. It was a 6x smaller one that beat the PCA benchmark in the 2022 bear market.
Along the way, the network discovered market sectors on its own, without ever seeing a sector label.

Read the full article below 👇

#MachineLearning #QuantitativeFinance #DeepLearning #FinTech

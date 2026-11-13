> **DRAFT written by Claude on 2026-11-12 as a scaffold for the author to rewrite in their own words.** Citations marked "verified" in `references.md` were confirmed by web search on that date; the others are from memory and must be checked. Read each source yourself before citing it: this draft only uses what the search results and standard knowledge support, but you should not cite a paper you have not opened. `[CHECK]` = verify; `[YOUR CALL]` = your judgment.

# 2. Background

## 2.1 Anomaly detection with autoencoders

Anomaly detection is the problem of finding observations that do not conform to expected behavior. Chandola, Banerjee, and Kumar (2009) give a structured overview, grouping techniques by their underlying approach and by the assumptions each makes about how normal and anomalous data differ.

An autoencoder is a neural network trained to reproduce its input after passing it through a narrow intermediate representation, a bottleneck (Hinton & Salakhutdinov, 2006). Trained on data that is mostly normal, it learns to reconstruct normal patterns well, so a large reconstruction error suggests an input unlike the training data. Sakurada and Yairi (2014) used autoencoders with nonlinear dimensionality reduction for anomaly detection and compared them with principal component analysis. For sequences, the recurrent LSTM network (Hochreiter & Schmidhuber, 1997) can be used for both encoder and decoder. Malhotra et al. (2016) trained an LSTM encoder-decoder only on normal time series and used reconstruction error as the anomaly score. In finance, Schreyer et al. (2017) applied deep autoencoders to accounting journal entries to flag suspicious ones.

Two design choices matter in what follows. First, the bottleneck must be small enough that the network cannot simply copy its input; otherwise reconstruction error stops measuring how unusual an input is. Second, the score is a number that must be turned into a flag with a threshold, and a common choice is a high percentile of errors on normal data, which fixes the false-positive rate by construction rather than measuring it.

## 2.2 Evaluating detectors without labels

Real labeled anomalies are scarce, and the methods surveyed by Chandola et al. (2009) differ partly in what labels they assume. A common workaround is to inject synthetic anomalies into normal data and measure how many are caught `[CHECK: find a citation for synthetic injection as standard practice]`. The workaround is only as good as the injected anomalies are realistic. If an injected anomaly could never occur in real data, a detector can score well by detecting impossibility rather than unusual behavior. Section 4.1 shows this happened in the earlier version of this project, and it is why evaluation receives so much attention here.

## 2.3 Faithfulness, hallucination, and grounding in generated text

Language models can state things that the input does not support. Ji et al. (2023) survey this problem, called hallucination, across natural language generation tasks, and Maynez et al. (2020) found in a large human evaluation of neural summarization systems that models often generate content unfaithful to the source and that overlap-based metrics are a weak proxy for faithfulness. For numbers in particular, a fluent sentence can contain a plausible but wrong figure.

One response is to remove the model's chance to invent facts. Here, every quantity in an explanation is computed in ordinary code and handed to the model, which only supplies wording, and the generated text is then checked against the same facts. This is a design rationale, not a result from the cited literature.

Evaluating generated text is itself difficult. A practical option is to use a strong language model as the judge. Zheng et al. (2023) studied this, reported that GPT-4 as judge agreed with human preferences more than 80% of the time on chat-assistant answers, and documented biases such as position and verbosity bias. That evidence concerns open-ended chat preferences, not a rubric about numeric faithfulness, so it does not transfer automatically to the setting here (Section 3.8).

## 2.4 Small open models

Running a model on the user's own machine keeps financial data off third-party servers and removes per-request cost. The explanation model used here is Llama 3.2 with 3 billion parameters, from the Llama 3 family of open models (Dubey et al., 2024) `[CHECK: confirm the best citation for the 3.2 release]`, served locally with Ollama. A model this small is much weaker than frontier models, which matters for the results in Section 4.8.

## 2.5 Freelancer finances and estimated tax

In the United States, people in business for themselves generally pay income tax through estimated payments made in installments across the year rather than through withholding, and the year is divided into four payment periods (Internal Revenue Service, n.d., Publication 505). This motivates a "quarterly set-aside" idea: putting aside a share of income as it arrives. The dataset here, however, appears to be in rupees, and this thesis did not research Indian tax rules. The tax module (Section 3.9) is therefore a flat-rate placeholder, not a calculation. `[YOUR CALL: add a source for the relevant tax rules, or keep this section generic and say so]`

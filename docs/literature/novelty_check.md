# Novelty check: evidence-adaptive sparse fusion (HUG, WWW'27 draft)

Date: 2026-10-06. Scope: the fusion contribution in `paper/main.tex` §"Evidence-adaptive sparse fusion" and the matching claims in the Introduction and Related Work. Sources: web search, arXiv, Crossref/ACM/IEEE DOI metadata. Every paper listed below was actually located; each has a URL.

## 1. Summary verdict

**No single paper we found pre-empts the full combination.** The full combination is: graph view + sequence view for CTR, fused through sparse codes over a shared dictionary plus view-private dictionaries, with an L0 / top-k budget that depends on per-impression evidence, and a reported shared-mass ratio. **Every ingredient has close prior work, though**, and three claims in the current draft are too strong.

| Claim component | Status | Closest prior work |
|---|---|---|
| Shared/private (invariant/specific) decomposition of two views in recommendation | **Not novel in itself.** Well established for modalities and domains, and now for semantic and collaborative signals (KDD'26). We found no paper that does it for a *graph view vs. sequence view* in CTR. | RIDRec (KDD'26), CMDL (TOIS'25), DisenCDR (SIGIR'22), MRdIB (arXiv'25) |
| Sparse codes over shared + private dictionaries | **Not novel as a technique** (Jia et al., NeurIPS'10; group-sparse SAEs, 2026). **Apparently new as a trained fusion layer in a recommender.** Rec-side SAE work is post-hoc interpretation, and rec-side discrete codes are tokenisers. | Jia/Salzmann/Darrell 2010; Kaushik et al. 2026; Klenitskiy et al. 2025; TIGER; MoToRec |
| Evidence-adaptive capacity (low evidence leads to less or different representation) | **Not novel as a principle.** Frequency-dependent embedding size is a mature line (MDE, AutoEmb, PEP, DESS). A *user-dependent information-bottleneck channel* whose bandwidth depends on how sufficient the collaborative evidence is already exists (VBAE). Maturity-gated fusion and alignment between two signal sources exists too (GateSID). | VBAE (TKDE'22), GateSID (arXiv'26), MDE (ISIT'21), AutoEmb (ICDM'21), PEP (ICLR'21) |
| Evidence-dependent L0 penalty *specifically on private atoms*, so low-evidence impressions fall back to shared atoms | **Not found. This is the defensible core.** | (none found) |
| Shared-atom mass ρ as a per-prediction measure of view overlap, analysed by evidence bucket | **Partly new.** RIDRec already decomposes predictive information into shared and unique parts. A per-prediction, countable overlap measure, sliced by evidence, was not found. | RIDRec; PID literature |
| Graph + sequence fusion "is typically concatenation, summation or attention" | **Needs softening.** Adaptive gating between a graph view and a sequence view exists (SGFRec). Adaptive gating conditioned on interaction sparsity exists for other view pairs (GateSID, VisGate). | SGFRec, GateSID, VisGate, MVCrec |

**Bottom line:** the paper can claim novelty for a *specific mechanism*: an evidence-conditioned sparsity budget acting asymmetrically on private versus shared dictionary atoms, used as the fusion layer between a graph encoder and a sequence encoder in CTR, together with the overlap measurement it enables. It should **not** claim novelty for (i) separating shared from view-specific information, (ii) using sparse or dictionary codes for multi-view fusion, or (iii) tying representation capacity to evidence or popularity.

Reviewers will want baselines that rule out cheaper explanations. Section 5 gives the list.

---

## 2. Papers found, grouped by topic

Verdict key: **pre-empts** / **close: must cite and differentiate** / **related: cite**.

### (a) Shared/private decomposition of views in recommendation / CTR

1. **J. Xia, Y. Yang, X. Wang, N. Liu. "Decomposing Predictive Roles of Semantic and Collaborative Information for Sequential Recommendation" (RIDRec).** KDD 2026 (Proc. 32nd ACM SIGKDD, V.2), pp. 5591–5601. https://doi.org/10.1145/3770855.3818088
   Splits next-item predictive information into three parts: *shared*, *semantic-unique* and *collaborative-unique*. It learns a role-specific representation for each with alignment and separation constraints, and argues explicitly against concatenation or joint encoding. This is the same philosophical argument as HUG's Introduction ("keep views separate, separate redundant from complementary evidence"). The view pair is different: LLM/semantic vs. collaborative, not graph vs. sequence. The representations are dense, not sparse codes, and there is no evidence-adaptive budget. **Verdict: close — must cite and differentiate.** It is the most recent and most direct conceptual precedent, and it appeared at a top venue two months ago.

2. **X. Lin, R. Liu, Y. Cao, L. Zou, Q. Li, Y. Wu, Y. Liu, D. Yin, G. Xu. "Contrastive Modality-Disentangled Learning for Multimodal Recommendation" (CMDL).** ACM TOIS 43(3):1–31, 2025. https://doi.org/10.1145/3715876
   Disentangles each modality into modality-invariant and modality-specific representations. Contrastive MI upper bounds enforce statistical independence. The motivation matches HUG's: a shared part to align views, a unique part for what only one view knows. This is MISA transplanted to recommendation. The views are modalities, the representations are dense, and nothing adapts to evidence. **Verdict: close — must cite and differentiate.** It directly undercuts "MISA-style decomposition has not been applied to recommenders".

3. **J. Cao, X. Lin, X. Cong, J. Ya, T. Liu, B. Wang. "DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation."** SIGIR 2022, pp. 267–277. https://doi.org/10.1145/3477495.3531967
   A VAE with domain-shared and domain-specific user embeddings and two MI regularisers. It fuses only the shared part across domains. This is the canonical shared/specific decomposition in recommendation, but with domains as the views. **Verdict: related — cite.**

4. **H. Wang, J. Qin, W. Wen, Q. Li, S. Zhong, Z. Huang. "Multimodal Representation-disentangled Information Bottleneck for Multimodal Recommendation" (MRdIB).** arXiv:2509.20225, 2025. https://arxiv.org/abs/2509.20225
   Applies an IB to compress, then decomposes multimodal information into unique, redundant and synergistic parts (PID-style) for recommendation. Overlap: the IB framing and explicit redundancy-vs-complementarity decomposition. Differences: modalities rather than graph/sequence, no sparse dictionary, no per-sample rate. **Verdict: close — must cite and differentiate**, especially because HUG invokes an IB reading of λ(n).

5. **X. Zhou, K. Lee. "ID and Graph View Contrastive Learning with Multi-View Attention Fusion for Sequential Recommendation" (MVCrec).** arXiv:2604.14114 (author PDF labelled IEEE BigData 2024). https://arxiv.org/abs/2604.14114
   An ID/sequence view and a graph view, with intra-view and cross-view contrastive losses and a multi-view attention fusion. It does **not** decompose into shared and specific parts. **Verdict: related — cite** (group e).

6. **J. Li et al. "Disentangling Multiplex Spatial-Temporal Transition Graph Representation Learning for Socially Enhanced POI Recommendation" (DiMuST).** arXiv:2508.07649, 2025 (**withdrawn** Oct 2025). https://arxiv.org/abs/2508.07649
   A disentangled variational graph autoencoder with shared and private distributions across spatial and temporal graph views. Shared parts are fused by product-of-experts and private parts are contrastively denoised. **Verdict: related.** Because the paper is withdrawn, mention it only if needed.

### (b) Sparse coding / dictionaries / SAEs / discrete codes in or around recommendation

7. **Y. Jia, M. Salzmann, T. Darrell. "Factorized Latent Spaces with Structured Sparsity."** NeurIPS 2010. https://home.ttic.edu/~salzmann/papers/JiaSalzmannDarrellNIPS10.pdf (also https://www2.eecs.berkeley.edu/Pubs/TechRpts/2010/EECS-2010-99.html)
   Multi-view sparse coding in which structured sparsity makes some dictionary atoms shared across views and others private to a single view. **This is the direct technical ancestor of HUG's "shared dictionary + view-private dictionaries".** It comes from vision (pose estimation), not recommendation, and it has no evidence adaptivity. **Verdict: close — must cite and differentiate.** Without it, reviewers will say the decomposition is a known technique presented as new.

8. **C. Kaushik, D. Barch, A. Fanelli. "Decomposing multimodal embedding spaces with group-sparse autoencoders"** (ICLR 2026 poster titled "Learning multimodal dictionary decompositions with group-sparse autoencoders"). arXiv:2601.20028. https://arxiv.org/abs/2601.20028 ; https://iclr.cc/virtual/2026/poster/10008818
   Shows that SAEs trained on aligned multimodal embeddings learn "split dictionaries" (mostly unimodal atoms). Uses cross-modal masking and group sparsity to force paired samples to share sparse support. This bears directly on whether HUG's shared atoms will actually be shared, and it suggests a failure mode to monitor: shared atoms collapsing into private ones. **Verdict: related — cite** (supports the design and the diagnostic).

9. **A. Klenitskiy et al. "Sparse Autoencoders for Sequential Recommendation Models: Interpretation and Flexible Control."** arXiv:2507.12202, 2025. https://arxiv.org/abs/2507.12202
   Trains an SAE *post hoc* on a transformer sequential recommender's hidden states to get interpretable, steerable features. It is not a fusion layer and not trained jointly. **Verdict: related — cite.** It shows HUG's difference: SAE-style codes trained end-to-end as the fusion representation.

10. **J. Liu, Z. Zhang, R. C. C. Cheung. "MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation."** AAAI 2026. https://arxiv.org/abs/2602.11062
    A sparsity-regularised RQ-VAE produces discrete, shared tokens across modalities. It adds "adaptive rarity amplification" for cold-start items and fuses with collaborative signal through a graph encoder. Overlaps: sparse codes, fusion with collaborative signal, cold-start-aware weighting. Differences: shared tokens only (no private dictionaries), and rarity changes *loss weighting* rather than the code budget. **Verdict: related — cite.**

11. **S. Rajput et al. "Recommender Systems with Generative Retrieval" (TIGER).** NeurIPS 2023. Already cited. **Related.**

12. **Sparse-MoE CTR with per-example sparsity: Y. Yan, L. Li. "AdaEnsemble: Learning Adaptively Sparse Structured Ensemble Network for Click-Through Rate Prediction."** arXiv:2301.08353, 2023. https://arxiv.org/abs/2301.08353
    Per-example sparse expert routing plus per-example depth selection in CTR. Sample-adaptive sparsity exists in CTR, but it is not tied to an evidence count and not used for shared/private fusion. **Verdict: related — cite (optional).**

13. **AdaptiveK SAE: "AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations."** arXiv:2508.17320 (ACL 2026 Findings). https://arxiv.org/abs/2508.17320
    A top-k SAE whose k depends on a per-input complexity signal. This is the technique closest to HUG's alternative "top-k(n) schedule", from LLM interpretability rather than recommendation. **Verdict: related — cite if the top-k(n) variant is kept.**

### (c) Evidence-, popularity-, frequency- or uncertainty-adaptive capacity in recommendation

14. **Y. Zhu, Z. Chen. "Variational Bandwidth Auto-encoder for Hybrid Recommender Systems" (VBAE).** IEEE TKDE, 2022 (early access), https://doi.org/10.1109/TKDE.2022.3155408 ; arXiv:2105.07597 https://arxiv.org/abs/2105.07597
    Fuses a user's collaborative latent and feature latent through a "virtual communication channel" with a **user-dependent bandwidth**. The bandwidth is inferred from the uncertainty of the collaborative (rating) embedding: users with insufficient collaborative evidence get more information from the auxiliary view, and others get less. It is framed information-theoretically (IB). **This is the closest precedent for "evidence-adaptive fusion as a per-sample rate constraint".** Differences: two dense Gaussian latents (no shared/private dictionaries, no sparse codes); bandwidth is inferred from latent uncertainty rather than set from an explicit evidence count; collaborative and feature views rather than graph and sequence; top-N recommendation rather than CTR. **Verdict: close — must cite and differentiate.** The current sentence "λ(n) acts as a per-sample rate constraint" reads as new without it.

15. **H. Zhu, Y. Yu, L. Shen, B. Wang, X. Zeng. "GateSID: Adaptive Gating for Balancing Semantic and Collaborative Signals in Recommendation."** arXiv:2603.22916, 2026. https://arxiv.org/abs/2603.22916
    A gate driven by *item maturity* (embeddings plus statistical features) balances semantic-ID and collaborative signals. A gate-regulated contrastive alignment is **stronger for cold items and relaxed for popular ones**. That is exactly "low-evidence predictions rely on the shared or aligned part". Differences: dense gating and alignment weighting rather than sparse shared/private codes, and a different view pair. **Verdict: close — must cite and differentiate.** It is also a natural strong baseline: an evidence-conditioned gate between the graph and sequence views.

16. **A. A. Ginart, M. Naumov, D. Mudigere, J. Yang, J. Zou. "Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems" (MDE).** IEEE ISIT 2021, pp. 2786–2791. https://doi.org/10.1109/ISIT45174.2021.9517710 ; arXiv:1909.11810
    Embedding dimension scales with query frequency (popularity), evaluated on Criteo CTR. This establishes "capacity proportional to evidence" for recommendation. **Verdict: close — must cite and differentiate.** HUG's twist is to adapt the *fusion code budget per impression*, and to adapt it asymmetrically for private versus shared atoms, instead of fixing embedding size per ID.

17. **X. Zhao et al. "AutoEmb: Automated Embedding Dimensionality Search in Streaming Recommendations."** IEEE ICDM 2021, pp. 896–905. https://doi.org/10.1109/ICDM51629.2021.00101 ; arXiv:2002.11252
    Differentiable selection of embedding dimension per user and item according to their (changing) popularity. **Verdict: close — must cite** (together with MDE as the "frequency-aware capacity" line).

18. **S. Liu, C. Gao, Y. Chen, D. Jin, Y. Li. "Learnable Embedding Sizes for Recommender Systems" (PEP).** ICLR 2021. https://arxiv.org/abs/2101.07577
    Learns pruning thresholds, giving mixed per-feature embedding sizes. A sparsity-based, learned capacity allocation, but not conditioned on evidence at inference time. **Verdict: related — cite.**

19. **Dynamic embedding size for streaming recommendation (DESS):** "Dynamic Embedding Size Search with Minimum Regret for Streaming Recommender System." arXiv:2308.07760 (CIKM 2023). https://arxiv.org/abs/2308.07760 . **Related — optional cite.**

20. **N. Glisovic, D. Kragic, M. Tegner. "Deciding When to Rely on Visual Information: Gated Multimodal Fusion in Sequential Recommendation" (VisGate).** CARS workshop @ RecSys 2026. https://arxiv.org/abs/2608.10700
    A learned gate shows that the utility of the visual view *increases under interaction sparsity*. This is an empirical finding parallel to HUG's motivation ("the structural view matters most for long-tail items"). Workshop paper. **Verdict: related — cite.**

21. **K. Kim, D. Hyun, S. Yun, C. Park. "MELT: Mutual Enhancement of Long-Tailed User and Item for Sequential Recommendation."** SIGIR 2023. https://arxiv.org/abs/2304.08382 . **Related — cite** in the long-tail motivation.

22. **LLM-ESR: "Large Language Models Enhancement for Long-tailed Sequential Recommendation."** NeurIPS 2024. https://neurips.cc/virtual/2024/poster/93061
    Dual-view (semantic + collaborative) modelling aimed at long-tail items. **Related — cite.**

### (d) Information bottleneck / multi-view IB / MISA-style in recommendation

23. **C. Wei, J. Liang, D. Liu, F. Wang. "Contrastive Graph Structure Learning via Information Bottleneck for Recommendation" (CGI).** NeurIPS 2022. https://papers.neurips.cc/paper_files/paper/2022/file/803b9c4a8e4784072fdd791c54d614e2-Paper-Conference.pdf
    An IB inside graph contrastive learning, to stop views capturing irrelevant information and to counter popularity bias. **Related — cite.**

24. **J. Cao, J. Sheng, X. Cong, T. Liu, B. Wang. "Cross-Domain Recommendation to Cold-Start Users via Variational Information Bottleneck" (CDRIB).** ICDE 2022, pp. 2209–2223. https://doi.org/10.1109/ICDE53745.2022.00211
    VIB regularisers force representations to keep only domain-shared, predictive information for cold-start users. This is the closest rec analogue of multi-view IB (Federici et al.). **Related — cite.**

25. MRdIB (item 4) and VBAE (item 14) also belong here. CMDL (item 2) is the MISA-style recommender.

### (e) Graph + sequence fusion, 2022–2026

26. **Y. Yang, C. Huang, L. Xia, C. Huang, D. Luo, K. Lin. "Debiased Contrastive Learning for Sequential Recommendation" (DCRec).** WWW 2023, pp. 1063–1073. https://doi.org/10.1145/3543507.3583361
    Unifies sequential-pattern encoding with a global collaborative graph view through cross-view contrastive learning, with *adaptive, conformity (popularity)-aware* augmentation weights. It is a WWW paper combining graph and sequence views with popularity-adaptive weighting, so WWW reviewers will know it. **Verdict: close — must cite and differentiate.** Its adaptivity acts on contrastive augmentation, not on the fusion representation's capacity.

27. **SGFRec: "Personalized Course Recommendation Based on Sequence Graph Fusion."** Applied Sciences 16(16):8078, 2026. https://www.mdpi.com/2076-3417/16/16/8078
    A sequence view (recent sequence) and a graph view (interaction graph), fused by a **learner-adaptive representation-level gate** whose weights depend on each user's sequential and graph representations. It is a direct counter-example to "fusion is typically concatenation, summation or attention". Low-tier venue. **Verdict: related — cite** (and soften that sentence).

28. **X. Liu, Z. Xiao, L. Yang, H. Xue, J. Ma, Y. Yang. "Improving CTR Prediction with Graph-Enhanced Interest Networks for Sparse Behavior Sequences."** WSDM 2025, pp. 876–884. https://doi.org/10.1145/3701551.3703567
    CTR with graph-enhanced interest modelling, targeted at users with sparse behaviour sequences. This is the same "graph helps when the sequence is thin" motivation in CTR. The abstract could not be retrieved (ACM DL returned 403), so overlap details are unverified. **Verdict: related — cite** (check the full text for its fusion design).

29. **Y. Liu, L. Xia, C. Huang. "SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation."** SIGIR 2024, pp. 1609–1618. https://doi.org/10.1145/3626772.3657716 . **Related — cite.**

30. **Y. Ye, L. Xia, C. Huang. "Graph Masked Autoencoder for Sequential Recommendation" (MAERec).** SIGIR 2023. https://arxiv.org/abs/2305.04619 . **Related — optional.**

31. Already cited and confirmed relevant: SURGE (SIGIR'21), GCL4SR (IJCAI'22), MCLSR (CIKM'22), GCE-GNN (SIGIR'20). MVCrec (item 5) also belongs here.

Searches that returned **nothing** matching: (i) shared/private decomposition between a *graph* view and a *sequence* view in CTR or sequential recommendation; (ii) a dictionary or SAE layer trained end-to-end as the *fusion* layer of a recommender; (iii) an L0/hard-concrete or top-k budget conditioned on an item- or user-evidence count in recommendation; (iv) a per-prediction "shared fraction" reported by evidence bucket. Absence from search is not proof of absence. Before submission, rerun searches on ACM DL full text for "shared atoms", "private dictionary", and "hard concrete recommendation".

---

## 3. Specific problems in the current draft text

- Related Work, "Combining graphs and sequences": *"Fusion is typically concatenation, summation or attention."* This is no longer accurate. Add gating (SGFRec), evidence- or maturity-conditioned gating (GateSID, VisGate) and popularity-adaptive cross-view contrast (DCRec).
- Related Work, "Multi-view and multimodal decomposition": MISA-style decomposition **has** been applied to recommenders (CMDL, DisenCDR, RIDRec, MRdIB). Say that, then differentiate.
- The claim *"sparse over learned dictionaries, so shared and private evidence can be counted"* should cite Jia et al. (2010) as the origin of shared/private sparse dictionaries. The new part is the recommender setting and the evidence-conditioned budget.
- The claim *"Its sparsity budget is evidence-adaptive, tying representation capacity to how much the model knows"* must cite MDE, AutoEmb (capacity proportional to frequency) and VBAE (user-dependent IB bandwidth from evidence sufficiency). The new part is narrower: the budget acts *per impression*, *asymmetrically on private atoms*, and *inside the fusion layer*.
- "Evidence" collides with *evidential deep learning* (Dirichlet "evidence", trusted multi-view classification). Define n as an interaction-count proxy early, to avoid reviewer confusion.
- The `\note{Run a targeted literature search ...}` in Related Work can now be resolved with the citations above.

## 4. Suggested novelty statement (2–3 sentences)

> Prior work separates shared from view-specific information with dense representations (MISA-style decomposition in multimodal and cross-domain recommendation, and recently semantic vs. collaborative signals). Prior work also adapts representation capacity to popularity, through per-ID embedding sizes or a user-dependent information channel. HUG combines the two inside the fusion layer between a graph encoder and a sequence encoder for CTR. Each view is encoded as sparse codes over a shared and a view-private dictionary, and the penalty on *private* atoms is set per impression from the candidate's and user's interaction evidence, so poorly supported predictions are forced onto atoms that both views agree on. Because the codes are sparse, the same mechanism yields a per-prediction, countable measure of how much the two views overlap, which we analyse across evidence levels.

Shorter contribution-bullet version: *"An evidence-conditioned sparse fusion layer that decomposes graph and sequence views into shared and private dictionary codes and charges private codes more when evidence is scarce. To our knowledge, this is the first fusion layer for CTR whose shared/private allocation is tied to per-impression evidence."* Keep "to our knowledge" and the narrow scope.

## 5. Baselines and ablations reviewers will ask for (given this prior work)

1. **Evidence-conditioned gate** between e^G and e^S (GateSID/SGFRec-style: a gate MLP on [e^G, e^S, log(1+n)]). This is the cheapest alternative explanation for gains concentrated on low-evidence items.
2. **Dense shared/private** (MISA/CMDL-style: same align + HSIC losses, no sparsity, no budget). This isolates the value of sparsity.
3. **Sparse shared/private with a constant λ** (no evidence dependence). This isolates the value of λ(n).
4. **Frequency-dependent capacity on the inputs** (MDE-style dimension proportional to popularity) as a capacity-allocation baseline.
5. Monitor **split-dictionary collapse** (Kaushik et al. 2026): report the share of atoms that are actually co-activated by both views.

## 6. BibTeX for the "must cite" papers

```bibtex
% verified: https://doi.org/10.1145/3770855.3818088
@inproceedings{xia2026ridrec,
  author    = {Xia, Jiangnan and Yang, Yu and Wang, Xiang and Liu, Ninghao},
  title     = {Decomposing Predictive Roles of Semantic and Collaborative Information for Sequential Recommendation},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  pages     = {5591--5601},
  doi       = {10.1145/3770855.3818088}
}

% verified: https://doi.org/10.1145/3715876
@article{lin2025cmdl,
  author  = {Lin, Xixun and Liu, Rui and Cao, Yanan and Zou, Lixin and Li, Qian and Wu, Yongxuan and Liu, Yang and Yin, Dawei and Xu, Guandong},
  title   = {Contrastive Modality-Disentangled Learning for Multimodal Recommendation},
  journal = {ACM Transactions on Information Systems},
  volume  = {43},
  number  = {3},
  pages   = {1--31},
  year    = {2025},
  doi     = {10.1145/3715876}
}

% verified: https://arxiv.org/abs/2509.20225
@article{wang2025mrdib,
  author  = {Wang, Hui and Qin, Jinghui and Wen, Wushao and Li, Qingling and Zhong, Shanshan and Huang, Zhongzhan},
  title   = {Multimodal Representation-disentangled Information Bottleneck for Multimodal Recommendation},
  journal = {arXiv preprint arXiv:2509.20225},
  year    = {2025}
}

% verified: https://home.ttic.edu/~salzmann/papers/JiaSalzmannDarrellNIPS10.pdf
@inproceedings{jia2010factorized,
  author    = {Jia, Yangqing and Salzmann, Mathieu and Darrell, Trevor},
  title     = {Factorized Latent Spaces with Structured Sparsity},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {23},
  year      = {2010}
}

% verified: https://doi.org/10.1109/TKDE.2022.3155408 (early access; final volume/pages not checked)
@article{zhu2022vbae,
  author  = {Zhu, Yaochen and Chen, Zhenzhong},
  title   = {Variational Bandwidth Auto-encoder for Hybrid Recommender Systems},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2022},
  doi     = {10.1109/TKDE.2022.3155408}
}

% verified: https://arxiv.org/abs/2603.22916
@article{zhu2026gatesid,
  author  = {Zhu, Hai and Yu, Yantao and Shen, Lei and Wang, Bing and Zeng, Xiaoyi},
  title   = {{GateSID}: Adaptive Gating for Balancing Semantic and Collaborative Signals in Recommendation},
  journal = {arXiv preprint arXiv:2603.22916},
  year    = {2026}
}

% verified: https://doi.org/10.1109/ISIT45174.2021.9517710
@inproceedings{ginart2021mde,
  author    = {Ginart, Antonio A. and Naumov, Maxim and Mudigere, Dheevatsa and Yang, Jiyan and Zou, James},
  title     = {Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems},
  booktitle = {2021 IEEE International Symposium on Information Theory (ISIT)},
  year      = {2021},
  pages     = {2786--2791},
  doi       = {10.1109/ISIT45174.2021.9517710}
}

% verified: https://doi.org/10.1109/ICDM51629.2021.00101
@inproceedings{zhao2021autoemb,
  author    = {Zhao, Xiangyu and Liu, Haochen and Fan, Wenqi and Liu, Hui and Tang, Jiliang and Wang, Chong and Chen, Ming and Zheng, Xudong and Liu, Xiaobing and Yang, Xiwang},
  title     = {{AutoEmb}: Automated Embedding Dimensionality Search in Streaming Recommendations},
  booktitle = {2021 IEEE International Conference on Data Mining (ICDM)},
  year      = {2021},
  pages     = {896--905},
  doi       = {10.1109/ICDM51629.2021.00101}
}

% verified: https://doi.org/10.1145/3543507.3583361
@inproceedings{yang2023dcrec,
  author    = {Yang, Yuhao and Huang, Chao and Xia, Lianghao and Huang, Chunzhen and Luo, Da and Lin, Kangyi},
  title     = {Debiased Contrastive Learning for Sequential Recommendation},
  booktitle = {Proceedings of the ACM Web Conference 2023},
  year      = {2023},
  pages     = {1063--1073},
  doi       = {10.1145/3543507.3583361}
}
```

Recommended "related — cite" entries, with keys suggested and metadata verified as noted:

```bibtex
% verified: https://doi.org/10.1145/3477495.3531967
@inproceedings{cao2022disencdr,
  author    = {Cao, Jiangxia and Lin, Xixun and Cong, Xin and Ya, Jing and Liu, Tingwen and Wang, Bin},
  title     = {{DisenCDR}: Learning Disentangled Representations for Cross-Domain Recommendation},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2022},
  pages     = {267--277},
  doi       = {10.1145/3477495.3531967}
}

% verified: https://doi.org/10.1109/ICDE53745.2022.00211
@inproceedings{cao2022cdrib,
  author    = {Cao, Jiangxia and Sheng, Jiawei and Cong, Xin and Liu, Tingwen and Wang, Bin},
  title     = {Cross-Domain Recommendation to Cold-Start Users via Variational Information Bottleneck},
  booktitle = {2022 IEEE 38th International Conference on Data Engineering (ICDE)},
  year      = {2022},
  pages     = {2209--2223},
  doi       = {10.1109/ICDE53745.2022.00211}
}

% verified: https://arxiv.org/abs/2101.07577
@inproceedings{liu2021pep,
  author    = {Liu, Siyi and Gao, Chen and Chen, Yihong and Jin, Depeng and Li, Yong},
  title     = {Learnable Embedding Sizes for Recommender Systems},
  booktitle = {International Conference on Learning Representations},
  year      = {2021}
}

% verified: https://arxiv.org/abs/2601.20028
@article{kaushik2026groupsae,
  author  = {Kaushik, Chiraag and Barch, Davis and Fanelli, Andrea},
  title   = {Decomposing Multimodal Embedding Spaces with Group-Sparse Autoencoders},
  journal = {arXiv preprint arXiv:2601.20028},
  year    = {2026}
}

% verified: https://arxiv.org/abs/2604.14114
@article{zhou2026mvcrec,
  author  = {Zhou, Xiaofan and Lee, Kyumin},
  title   = {{ID} and Graph View Contrastive Learning with Multi-View Attention Fusion for Sequential Recommendation},
  journal = {arXiv preprint arXiv:2604.14114},
  year    = {2026}
}

% verified: https://doi.org/10.1145/3701551.3703567
@inproceedings{liu2025geinsparse,
  author    = {Liu, Xuanzhou and Xiao, Zhibo and Yang, Luwei and Xue, Hansheng and Ma, Jianxing and Yang, Yujiu},
  title     = {Improving {CTR} Prediction with Graph-Enhanced Interest Networks for Sparse Behavior Sequences},
  booktitle = {Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining},
  year      = {2025},
  pages     = {876--884},
  doi       = {10.1145/3701551.3703567}
}

% verified: https://doi.org/10.1145/3626772.3657716
@inproceedings{liu2024selfgnn,
  author    = {Liu, Yuxi and Xia, Lianghao and Huang, Chao},
  title     = {{SelfGNN}: Self-Supervised Graph Neural Networks for Sequential Recommendation},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2024},
  pages     = {1609--1618},
  doi       = {10.1145/3626772.3657716}
}
```

(Also consider: Klenitskiy et al. 2025 arXiv:2507.12202; MoToRec AAAI 2026 arXiv:2602.11062; VisGate arXiv:2608.10700; CGI NeurIPS 2022; MELT SIGIR 2023; LLM-ESR NeurIPS 2024; AdaptiveK SAE arXiv:2508.17320. Look up full author lists before adding them.)

---

## 7. Task 2: bibliography changelog (`paper/references.bib`)

All 41 entries were checked against Crossref DOI metadata, PMLR, arXiv, or ICLR/NeurIPS pages. Each entry now has a `% verified: <url>` line, and all `% VERIFY` markers were removed. DBLP blocks automated access (Anubis bot wall), so it was not used directly. **0 entries are unverified.** Citation keys are unchanged. `main.tex` builds with tectonic with no undefined citations (40 cited keys resolve to 40 bibitems).

**Substantive corrections (8 entries):**
- `yu2023xsimgcl`: year 2023 → **2024**; added vol. 36, no. 2, pp. 913–926, and the DOI (it was early access in 2023).
- `zhang2024wukong`: **author list was wrong**. Removed "Daifeng Guo", fixed the order (Li, Shen before Zhao, Yanli), and replaced "others" with the full PMLR list (… Yao, Wen, Park, Naumov, Chen). Added PMLR 235:59421–59434.
- `zhai2024hstu`: **author error**: "He, Michael" → **"He, Jiayuan"**. Completed the list (Lu, Yinghai; Shi, Yu). Added PMLR 235:58484–58509.
- `zhang2022oneepoch`: **title was wrong**. Corrected to "…Overfitting Phenomenon of Deep Click-Through Rate **Models**" (no "Prediction"). Added pp. 2671–2680.
- `gao2024sae`: arXiv preprint → **ICLR 2025** conference paper (key kept).
- `ji2023leakage`: added vol. 41, no. 3, pp. 1–27, and the DOI.
- `guo2024collapse`: authors confirmed; added PMLR 235:16891–16909.
- `wu2019srgnn`: added vol. 33 and the DOI. The `number` field was dropped because ACM-Reference-Format rejects volume+number together.

**Minor completions (no factual error):** added DOIs throughout. Added series and volume for `wu2019sgc` (PMLR 97), `schlichtkrull2018rgcn` (LNCS 10843) and `gretton2005hsic` (LNCS 3734). Added pp. 1–4 for `chen2019bst`, and vol. 36 with pp. 10299–10315 for `rajput2023tiger` (middle initial "Keshavan, Raghunandan H.").

**Confirmed correct as written:** he2020lightgcn, yu2022simgcl, wang2019kgat, hu2020hgt, lv2021simplehgn, frasca2020sign, li2016ggnn, kang2018sasrec, zhou2018din, xia2023transact, li2019fignn, wang2021dcnv2, ma2018mmoe, chang2021surge, wang2020gcegnn, zhang2022gcl4sr, wang2022mclsr, hazarika2020misa, federici2020mib, tishby2000ib, alemi2017vib, makhzani2014ksparse, louizos2018l0, oord2018infonce, dacrema2019progress, rendle2020ncf, zhu2021bars, gao2022kuairand.

Residual BibTeX warnings ("empty publisher/address" for ACM style, and missing pages for ICLR/arXiv entries) are cosmetic and were present before. `li2016ggnn` is defined but not cited in main.tex.

# Contributing to Medical Insurance Claims Multimodal RAG

Thank you for your interest in contributing! We welcome contributions to enhance multimodal ingestion, retrieval algorithms, evaluation datasets, and chatbot capabilities.

---

## 🛠️ Development Workflow

1. **Fork the Repository** on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/medical-insurance-claims-rag.git
   cd medical-insurance-claims-rag
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up virtual environment**:
   ```bash
   python -m venv medical_rag_chatbot/venv
   source medical_rag_chatbot/venv/bin/activate # or venv\Scripts\activate on Windows
   pip install -r medical_rag_chatbot/requirements.txt
   ```
5. **Run test verification and regression checks**:
   ```bash
   python medical_rag_chatbot/test_setup.py
   python medical_rag_chatbot/evaluation/regression_check.py
   ```
6. **Commit with conventional commit messages**:
   ```bash
   git commit -m "feat: add table header normalization for dental claims"
   ```
7. **Push to your branch and open a Pull Request**.

---

## 📋 Pull Request Guidelines

- Ensure no secrets or API keys are committed.
- Maintain test coverage and verify that `regression_check.py` passes without regressions.
- Preserve HIPAA privacy standards: ensure mock data uses synthetic identifiers.

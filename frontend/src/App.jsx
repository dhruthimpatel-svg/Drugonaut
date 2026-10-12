
import { useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function MetricCard({ label, value, description }) {
  return (
    <article className="property-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <p className="field-hint">{description}</p>
    </article>
  );
}

export default function App() {
  const [page, setPage] = useState("explorer");

  const [smiles, setSmiles] = useState("Cn1c(=O)c2c(ncn2C)n(C)c1=O");
  const [result, setResult] = useState(null);
  const [moleculeImage, setMoleculeImage] = useState("");
  const [similarMolecules, setSimilarMolecules] = useState([]);
  const [similarNote, setSimilarNote] = useState("");
  const [similarLoading, setSimilarLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [similarError, setSimilarError] = useState("");

  const [evaluation, setEvaluation] = useState(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");

  async function analyzeMolecule(event) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);
    setMoleculeImage("");
    setSimilarMolecules([]);
    setSimilarNote("");
    setSimilarError("");
    setSimilarLoading(false);

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Molecule analysis failed."
        );
      }

      setResult(data);

      const encodedSmiles = encodeURIComponent(data.smiles);
      setMoleculeImage(
        `${API_URL}/api/molecule-image?smiles=${encodedSmiles}`
      );

      setSimilarLoading(true);

      try {
        const similarResponse = await fetch(
          `${API_URL}/api/similar-molecules?smiles=${encodedSmiles}&limit=10`
        );

        const similarData = await similarResponse.json();

        if (!similarResponse.ok) {
          throw new Error(
            typeof similarData.detail === "string"
              ? similarData.detail
              : "Similar molecule search failed."
          );
        }

        setSimilarMolecules(similarData.results || []);
        setSimilarNote(similarData.note || "");
      } catch (similarErr) {
        setSimilarError(
          similarErr.message.includes("Failed to fetch")
            ? "Could not connect to the similar-molecule endpoint."
            : similarErr.message
        );
      } finally {
        setSimilarLoading(false);
      }
    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? "The Drugonaut API is not reachable. Check that FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  async function openEvaluation() {
    setPage("evaluation");

    if (evaluation || evaluationLoading) return;

    setEvaluationLoading(true);
    setEvaluationError("");

    try {
      const response = await fetch(`${API_URL}/api/model-evaluation`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Could not load model evaluation."
        );
      }

      setEvaluation(data);
    } catch (err) {
      setEvaluationError(
        err.message.includes("Failed to fetch")
          ? "Could not connect to the Drugonaut API. Check that FastAPI is running."
          : err.message
      );
    } finally {
      setEvaluationLoading(false);
    }
  }

  function openExplorer() {
    setPage("explorer");
  }

  const randomSplit = evaluation?.evaluation?.splits?.find(
    (split) => split.name === "Random split"
  );

  const scaffoldSplit = evaluation?.evaluation?.splits?.find(
    (split) => split.name === "Scaffold split"
  );

  const confusion = scaffoldSplit?.confusion_matrix?.matrix;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault();
            openExplorer();
          }}
        >
          <span className="brand-mark">D</span>
          <span>
            drugonaut<span className="brand-period">.</span>
          </span>
        </a>

        <p className="nav-label">WORKSPACE</p>

        <button
          type="button"
          className={`nav-item ${page === "explorer" ? "active" : ""}`}
          onClick={openExplorer}
        >
          âŒ˜ <span>Molecule explorer</span>
        </button>

        <button
          type="button"
          className={`nav-item ${page === "evaluation" ? "active" : ""}`}
          onClick={openEvaluation}
        >
          â—« <span>Model evaluation</span>
        </button>

        <div className="sidebar-bottom">
          <span className="status-dot" />
          <span>Research workspace</span>
          <small>LOCAL</small>
        </div>
      </aside>

      <section className="main-content">
        <header className="topbar">
          <span>
            MOLECULAR RESEARCH /{" "}
            {page === "explorer" ? "EXPLORER" : "MODEL EVALUATION"}
          </span>
          <span className="version">PROTOTYPE 01</span>
        </header>

        {page === "explorer" ? (
          <>
            <section className="hero">
              <p className="eyebrow">COMPUTATIONAL MOLECULAR RESEARCH</p>
              <h1>
                Explore molecules.
                <br />
                <span>Understand their potential.</span>
              </h1>
              <p className="hero-copy">
                Analyze molecular properties and explore a
                machine-learning prediction of bloodâ€“brain barrier
                permeability.
              </p>
            </section>

            <section className="workspace-grid">
              <article className="panel molecule-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">01 / INPUT</p>
                    <h2>Molecule explorer</h2>
                  </div>
                  <span className="panel-icon">âŒ¬</span>
                </div>

                <form onSubmit={analyzeMolecule}>
                  <label htmlFor="smiles">SMILES notation</label>
                  <textarea
                    id="smiles"
                    value={smiles}
                    onChange={(event) => setSmiles(event.target.value)}
                    placeholder="Enter a molecule's SMILES..."
                    required
                    rows={4}
                  />

                  <p className="field-hint">
                    Enter a valid SMILES string to analyze a molecule.
                  </p>

                  <button
                    className="primary-button"
                    type="submit"
                    disabled={loading}
                  >
                    {loading ? "Analyzing molecule..." : "Analyze molecule  â†’"}
                  </button>
                </form>

                <div className="notice">
                  <span>i</span>
                  <p>
                    Predictions are research estimates, not proof of drug
                    safety, efficacy, or bloodâ€“brain barrier passage.
                  </p>
                </div>
              </article>

              <article className="panel results-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">02 / OUTPUT</p>
                    <h2>Analysis results</h2>
                  </div>
                  <span className="result-status">
                    {result ? "COMPLETE" : "AWAITING INPUT"}
                  </span>
                </div>

                {error && <div className="error-message">{error}</div>}

                {!result && !error && (
                  <div className="empty-state">
                    <div className="molecule-symbol">âŒ¬</div>
                    <h3>Your next discovery starts here.</h3>
                    <p>
                      Submit a molecule to see its calculated properties
                      and model output.
                    </p>
                  </div>
                )}

                {result && (
                  <div className="results-content">
                    {moleculeImage && (
                      <div className="structure-card">
                        <span className="eyebrow">MOLECULAR STRUCTURE</span>
                        <div className="structure-image-wrapper">
                          <img
                            src={moleculeImage}
                            alt="2D chemical structure of the analyzed molecule"
                            className="molecule-image"
                          />
                        </div>
                        <p className="structure-smiles">{result.smiles}</p>
                      </div>
                    )}

                    <div className="prediction-card">
                      <span className="eyebrow">BBBP MODEL PREDICTION</span>
                      <h3>
                        {result.prediction?.label ?? "Prediction returned"}
                      </h3>
                      <p>
                        Raw model score:{" "}
                        {typeof result.prediction?.score === "number"
                          ? result.prediction.score.toFixed(4)
                          : "Not provided"}
                      </p>
                      <p className="field-hint">
                        This score is not a calibrated probability.
                      </p>
                    </div>

                    <h3 className="properties-title">Molecular properties</h3>
                    <div className="properties-grid">
                      {Object.entries(result.descriptors || {}).map(
                        ([key, value]) => (
                          <div className="property-card" key={key}>
                            <span>{key.replaceAll("_", " ")}</span>
                            <strong>{String(value)}</strong>
                          </div>
                        )
                      )}
                    </div>

                    <section className="similar-section">
                      <div className="panel-heading">
                        <div>
                          <p className="eyebrow">03 / DISCOVERY</p>
                          <h2>Similar molecules</h2>
                        </div>
                        <span className="panel-icon">âŒ•</span>
                      </div>

                      <p className="field-hint">
                        Compounds ranked by structural resemblance using
                        Morgan fingerprints and Tanimoto similarity.
                      </p>

                      {similarLoading && (
                        <div className="empty-state">
                          <p>
                            Searching your BBBP dataset for similar
                            compounds...
                          </p>
                        </div>
                      )}

                      {similarError && (
                        <div className="error-message">{similarError}</div>
                      )}

                      {!similarLoading &&
                        !similarError &&
                        similarMolecules.length === 0 && (
                          <div className="empty-state">
                            <p>No similar compounds were returned.</p>
                          </div>
                        )}

                      {!similarLoading && similarMolecules.length > 0 && (
                        <div className="similar-list">
                          {similarMolecules.map((compound, index) => (
                            <article
                              className="similar-card"
                              key={`${compound.smiles}-${index}`}
                            >
                              <div className="similar-card-top">
                                <div>
                                  <span className="eyebrow">
                                    MATCH {String(index + 1).padStart(2, "0")}
                                  </span>
                                  <h3>{compound.name}</h3>
                                </div>
                                <span className="similarity-score">
                                  {(compound.similarity * 100).toFixed(1)}%
                                </span>
                              </div>

                              <div className="similarity-track">
                                <div
                                  className="similarity-fill"
                                  style={{
                                    width: `${Math.max(
                                      0,
                                      Math.min(
                                        100,
                                        compound.similarity * 100
                                      )
                                    )}%`,
                                  }}
                                />
                              </div>

                              <p className="structure-smiles">
                                {compound.smiles}
                              </p>

                              <div className="similar-meta">
                                <span>
                                  Dataset label:{" "}
                                  {compound.dataset_label === null ||
                                  compound.dataset_label === undefined
                                    ? "Unavailable"
                                    : compound.dataset_label}
                                </span>
                                <button
                                  className="secondary-button"
                                  type="button"
                                  onClick={() => {
                                    setSmiles(compound.smiles);
                                    window.scrollTo({
                                      top: 0,
                                      behavior: "smooth",
                                    });
                                  }}
                                >
                                  Use molecule
                                </button>
                              </div>
                            </article>
                          ))}
                        </div>
                      )}

                      {similarNote && (
                        <p className="field-hint">{similarNote}</p>
                      )}
                    </section>

                    {result.disclaimer && (
                      <p className="field-hint">{result.disclaimer}</p>
                    )}
                  </div>
                )}
              </article>
            </section>
          </>
        ) : (
          <>
            <section className="hero">
              <p className="eyebrow">MODEL QUALITY / GENERALIZATION</p>
              <h1>
                Test the model.
                <br />
                <span>Understand its limits.</span>
              </h1>
              <p className="hero-copy">
                Compare recorded random-split and scaffold-split metrics
                to understand how evaluation design affects reported
                performance.
              </p>
            </section>

            {evaluationLoading && (
              <article className="panel">
                <div className="empty-state">
                  <h3>Loading evaluation results...</h3>
                  <p>Fetching model metrics from the Drugonaut API.</p>
                </div>
              </article>
            )}

            {evaluationError && (
              <article className="panel">
                <div className="error-message">{evaluationError}</div>
                <button
                  className="primary-button"
                  type="button"
                  onClick={() => {
                    setEvaluation(null);
                    openEvaluation();
                  }}
                >
                  Retry loading
                </button>
              </article>
            )}

            {evaluation && !evaluationLoading && (
              <>
                <article className="panel">
                  <div className="panel-heading">
                    <div>
                      <p className="eyebrow">01 / MODEL DETAILS</p>
                      <h2>{evaluation.model}</h2>
                    </div>
                    <span className="result-status">RECORDED METRICS</span>
                  </div>

                  <div className="properties-grid">
                    <MetricCard
                      label="Dataset"
                      value={evaluation.dataset.name}
                      description={`${evaluation.dataset.valid_molecules} valid molecules`}
                    />
                    <MetricCard
                      label="Fingerprint"
                      value={`${evaluation.fingerprint.bits}-bit`}
                      description={`${evaluation.fingerprint.type} Â· radius ${evaluation.fingerprint.radius}`}
                    />
                    <MetricCard
                      label="Scaffold groups"
                      value={evaluation.dataset.scaffold_groups}
                      description="Distinct scaffold groups recorded"
                    />
                    <MetricCard
                      label="Scaffold overlap"
                      value={evaluation.dataset.scaffold_overlap}
                      description="Shared scaffolds between train and test"
                    />
                  </div>

                  <p className="field-hint">
                    These metrics are previously recorded results. This
                    page does not rerun model training or evaluation.
                  </p>
                </article>

                <article className="panel">
                  <div className="panel-heading">
                    <div>
                      <p className="eyebrow">02 / PERFORMANCE</p>
                      <h2>Evaluation comparison</h2>
                    </div>
                  </div>

                  <div className="properties-grid">
                    {[
                      ["Accuracy", "accuracy"],
                      ["Balanced accuracy", "balanced_accuracy"],
                      ["ROC-AUC", "roc_auc"],
                    ].map(([label, key]) => (
                      <div className="property-card" key={key}>
                        <span>{label}</span>
                        <div className="metric-comparison">
                          <div>
                            <small>Random</small>
                            <strong>
                              {randomSplit
                                ? formatPercent(randomSplit[key])
                                : "â€”"}
                            </strong>
                          </div>
                          <div>
                            <small>Scaffold</small>
                            <strong>
                              {scaffoldSplit
                                ? formatPercent(scaffoldSplit[key])
                                : "â€”"}
                            </strong>
                          </div>
                        </div>

                        {randomSplit && scaffoldSplit && (
                          <p className="field-hint">
                            {(
                              (randomSplit[key] - scaffoldSplit[key]) *
                              100
                            ).toFixed(1)}{" "}
                            percentage-point difference
                          </p>
                        )}
                      </div>
                    ))}
                  </div>

                  <div className="comparison-block">
                    <h3>Accuracy</h3>
                    {[randomSplit, scaffoldSplit]
                      .filter(Boolean)
                      .map((split) => (
                        <div className="metric-bar-row" key={split.name}>
                          <span>{split.name}</span>
                          <div className="similarity-track">
                            <div
                              className="similarity-fill"
                              style={{
                                width: `${split.accuracy * 100}%`,
                              }}
                            />
                          </div>
                          <strong>{formatPercent(split.accuracy)}</strong>
                        </div>
                      ))}
                  </div>

                  <p className="field-hint">
                    {evaluation.interpretation}
                  </p>
                </article>

                {confusion && (
                  <article className="panel">
                    <div className="panel-heading">
                      <div>
                        <p className="eyebrow">03 / ERROR ANALYSIS</p>
                        <h2>Scaffold-split confusion matrix</h2>
                      </div>
                    </div>

                    <p className="field-hint">
                      Rows represent actual classes; columns represent
                      predicted classes. Class labels are 0 and 1.
                    </p>

                    <div className="confusion-matrix">
                      <div className="confusion-cell matrix-heading">
                        Actual â†“ / Predicted â†’
                      </div>
                      <div className="confusion-cell matrix-heading">
                        Predicted 0
                      </div>
                      <div className="confusion-cell matrix-heading">
                        Predicted 1
                      </div>

                      <div className="confusion-cell matrix-heading">
                        Actual 0
                      </div>
                      <div className="confusion-cell">
                        <strong>{confusion[0][0]}</strong>
                        <small>True negative</small>
                      </div>
                      <div className="confusion-cell">
                        <strong>{confusion[0][1]}</strong>
                        <small>False positive</small>
                      </div>

                      <div className="confusion-cell matrix-heading">
                        Actual 1
                      </div>
                      <div className="confusion-cell">
                        <strong>{confusion[1][0]}</strong>
                        <small>False negative</small>
                      </div>
                      <div className="confusion-cell">
                        <strong>{confusion[1][1]}</strong>
                        <small>True positive</small>
                      </div>
                    </div>

                    <p className="field-hint">
                      The matrix contains {confusion.flat().reduce(
                        (total, value) => total + value,
                        0
                      )} test examples.
                    </p>
                  </article>
                )}

                <article className="panel">
                  <p className="eyebrow">04 / INTERPRETATION</p>
                  <h2>Why scaffold splitting matters</h2>
                  <p>
                    A random split can place structurally related molecules
                    in both training and test sets. A scaffold split attempts
                    to separate groups based on their core molecular
                    frameworks, providing a different test of generalization
                    to less familiar structures.
                  </p>
                  <p className="field-hint">
                    Lower scaffold-split scores can indicate that
                    generalization is harder under this split. The
                    comparison alone does not prove the cause of the
                    performance difference.
                  </p>
                  <p className="field-hint">
                    {evaluation.disclaimer}
                  </p>
                </article>
              </>
            )}
          </>
        )}

        <footer className="page-footer">
          <span>DRUGONAUT Â· MOLECULAR RESEARCH PROTOTYPE</span>
          <span>Built for exploration, not clinical decisions.</span>
        </footer>
      </section>
    </main>
  );
}


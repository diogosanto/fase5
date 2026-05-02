import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "docs";
const outputPath = `${outputDir}/tracker_responsaveis_datathon.xlsx`;

const workbook = Workbook.create();
const tracker = workbook.worksheets.add("Plano de Acao");
const listas = workbook.worksheets.add("Listas");
const resumo = workbook.worksheets.add("Resumo");

const headers = [
  "Macroetapa",
  "Subetapa",
  "Status atual",
  "Prioridade",
  "Responsavel",
  "Prazo",
  "Status execucao",
  "% concluido",
  "Proxima acao",
  "Evidencia/arquivo",
  "Observacoes",
];

const rows = [
  ["Dados, Features e Modelo", "Ingestao ITBI", "Parcial/OK", "Media", "", "", "Nao iniciado", "", "Revisar scripts e documentar origem dos dados", "src/data", ""],
  ["Dados, Features e Modelo", "Limpeza e normalizacao", "Parcial/OK", "Media", "", "", "Nao iniciado", "", "Validar colunas, tipos e amostras do dataset limpo", "src/data/2clean_all.py", ""],
  ["Dados, Features e Modelo", "Feature engineering", "Parcial", "Alta", "", "", "Nao iniciado", "", "Remover vazamento por valor_m2 derivado do alvo", "src/features/build_features_minimal.py", ""],
  ["Dados, Features e Modelo", "Treinamento MLflow", "Parcial/OK", "Alta", "", "", "Nao iniciado", "", "Retreinar modelo sem vazamento e registrar parametros/tags", "src/training/train_mlflow.py", ""],
  ["Dados, Features e Modelo", "Promocao dev/test/prod", "Parcial", "Media", "", "", "Nao iniciado", "", "Adicionar criterio real de promocao por metrica", "src/training/promote_model.py", ""],
  ["Dados, Features e Modelo", "Validacao de modelo", "Fraca", "Alta", "", "", "Nao iniciado", "", "Criar validacao com holdout, faixa de predicao e metricas", "src/training/validate_model.py", ""],
  ["Dados, Features e Modelo", "DVC/pipeline reproduzivel", "Falta", "Alta", "", "", "Nao iniciado", "", "Criar dvc.yaml com etapas data/features/train/validate", "dvc.yaml", ""],

  ["API Produto", "/health", "OK", "Baixa", "", "", "Nao iniciado", "", "Manter endpoint e incluir em testes de integracao", "api/main.py", ""],
  ["API Produto", "/predict", "Parcial", "Alta", "", "", "Nao iniciado", "", "Trocar parametros soltos por JSON Pydantic realista", "api/main.py", ""],
  ["API Produto", "/chat", "Parcial/novo", "Alta", "", "", "Nao iniciado", "", "Validar fluxo com agente, erros e resposta estruturada", "api/main.py", ""],
  ["API Produto", "Schema Pydantic", "Parcial", "Alta", "", "", "Nao iniciado", "", "Criar schemas para predict e chat", "api/main.py", ""],
  ["API Produto", "Logs", "Parcial/OK", "Media", "", "", "Nao iniciado", "", "Padronizar logs de inferencia e chat", "api/main.py", ""],
  ["API Produto", "Prometheus instrumentation", "Parcial/OK", "Media", "", "", "Nao iniciado", "", "Validar endpoint /metrics e nomes das metricas", "api/main.py", ""],
  ["API Produto", "Tratamento de erro", "Parcial", "Alta", "", "", "Nao iniciado", "", "Adicionar respostas 4xx/5xx claras e validação de entrada", "api/main.py", ""],

  ["LLM + Agente + RAG", "LLM servido via API com quantizacao", "Parcial/Falta", "Alta", "", "", "Nao iniciado", "", "Decidir se sera LLM local quantizado ou justificar Gemini API", "src/agent/llm.py", ""],
  ["LLM + Agente + RAG", "Agente ReAct com >= 3 tools", "Parcial/Quase", "Alta", "", "", "Nao iniciado", "", "Validar escolhas de tools e respostas finais", "src/agent/react_agent.py", ""],
  ["LLM + Agente + RAG", "Tool rag_search", "Parcial", "Alta", "", "", "Nao iniciado", "", "Validar recuperacao de contexto e fontes", "src/agent/tools.py; src/rag", ""],
  ["LLM + Agente + RAG", "Tool price_estimator", "Parcial", "Alta", "", "", "Nao iniciado", "", "Alinhar campos com modelo corrigido", "src/agent/tools.py", ""],
  ["LLM + Agente + RAG", "Tool region_comparer", "Parcial", "Media", "", "", "Nao iniciado", "", "Validar bairros e metricas com dataset processado", "src/agent/tools.py", ""],
  ["LLM + Agente + RAG", "RAG documental", "Parcial", "Alta", "", "", "Nao iniciado", "", "Ampliar documentos de apoio e revisar chunking/retriever", "data/rag/raw; src/rag", ""],
  ["LLM + Agente + RAG", "Golden set", "OK inicial", "Media", "", "", "Nao iniciado", "", "Revisar 20 perguntas e expected_tool", "data/golden_set/real_estate_chat_golden_set.jsonl", ""],
  ["LLM + Agente + RAG", "Benchmark 3 configuracoes", "Falta", "Alta", "", "", "Nao iniciado", "", "Comparar k=1/k=3/k=5 ou modelos/configs diferentes", "evaluation/benchmark_agent.py", ""],
  ["LLM + Agente + RAG", "Script de build RAG", "Parcial/quebrado", "Alta", "", "", "Nao iniciado", "", "Corrigir scripts/build_rag_index.py desatualizado", "scripts/build_rag_index.py", ""],

  ["Avaliacao", "Acuracia de tool escolhida", "Falta", "Alta", "", "", "Nao iniciado", "", "Medir expected_tool vs tools_used no golden set", "evaluation", ""],
  ["Avaliacao", "RAGAS 4 metricas", "Falta", "Media", "", "", "Nao iniciado", "", "Criar avaliacao RAGAS ou justificar alternativa simples", "evaluation/ragas_eval.py", ""],
  ["Avaliacao", "LLM-as-judge", "Falta", "Media", "", "", "Nao iniciado", "", "Definir 3 criterios: corretude, utilidade, seguranca", "evaluation/llm_judge.py", ""],
  ["Avaliacao", "Relatorio de benchmark", "Falta", "Alta", "", "", "Nao iniciado", "", "Documentar resultados, configs e conclusao", "docs", ""],

  ["Observabilidade", "Prometheus", "Parcial", "Media", "", "", "Nao iniciado", "", "Validar scrape da API", "monitoring/prometheus/prometheus.yml", ""],
  ["Observabilidade", "Grafana provisioning", "Melhorou", "Media", "", "", "Nao iniciado", "", "Validar dashboard no docker-compose", "monitoring/grafana", ""],
  ["Observabilidade", "Alertas", "Parcial/quebrado", "Alta", "", "", "Nao iniciado", "", "Padronizar job_name: precificador-api vs api_precificador", "monitoring/prometheus/alertrules.yml", ""],
  ["Observabilidade", "Metricas de negocio", "Falta", "Media", "", "", "Nao iniciado", "", "Adicionar contadores para predicoes/chat/erros", "api/main.py", ""],
  ["Observabilidade", "Drift detection", "Falta", "Alta", "", "", "Nao iniciado", "", "Criar script Evidently e documento de uso", "monitoring/drifts", ""],

  ["Testes e CI/CD", "Testes unitarios", "Falta", "Alta", "", "", "Nao iniciado", "", "Criar testes de features e tools", "tests/unit", ""],
  ["Testes e CI/CD", "Testes integracao API", "Falta", "Alta", "", "", "Nao iniciado", "", "Testar /health, /predict e /chat", "tests/integration", ""],
  ["Testes e CI/CD", "Testes RAG/agente", "Falta", "Alta", "", "", "Nao iniciado", "", "Testar golden set e selecao de tool", "tests/integration", ""],
  ["Testes e CI/CD", "pytest/dev deps", "Falta", "Alta", "", "", "Nao iniciado", "", "Adicionar pytest, httpx compativel e dependencias dev", "pyproject.toml", ""],
  ["Testes e CI/CD", "GitHub Actions", "Falta", "Alta", "", "", "Nao iniciado", "", "Criar workflow lint/test/build", ".github/workflows/ci.yml", ""],

  ["Seguranca, Governanca e Docs", "README principal", "Falta", "Alta", "", "", "Nao iniciado", "", "Documentar execucao, arquitetura, treino e demo", "README.md", ""],
  ["Seguranca, Governanca e Docs", "Model Card", "Falta", "Alta", "", "", "Nao iniciado", "", "Documentar modelo, dados, metricas, riscos e limites", "docs/MODEL_CARD.md", ""],
  ["Seguranca, Governanca e Docs", "System Card", "Falta", "Alta", "", "", "Nao iniciado", "", "Documentar sistema completo API + ML + agente", "docs/SYSTEM_CARD.md", ""],
  ["Seguranca, Governanca e Docs", "LGPD", "Falta", "Media", "", "", "Nao iniciado", "", "Documentar dados publicos, privacidade e riscos", "docs/LGPD_PLAN.md", ""],
  ["Seguranca, Governanca e Docs", "OWASP LLM", "Falta", "Media", "", "", "Nao iniciado", "", "Mapear 5 ameacas e mitigacoes", "docs/OWASP_MAPPING.md", ""],
  ["Seguranca, Governanca e Docs", "Guardrails", "Falta", "Media", "", "", "Nao iniciado", "", "Criar validacao de input/output e politicas", "src/security/guardrails.py", ""],
  ["Seguranca, Governanca e Docs", "Red teaming", "Falta", "Media", "", "", "Nao iniciado", "", "Criar 5 cenarios adversariais e resultados", "docs/RED_TEAM_REPORT.md", ""],
  ["Seguranca, Governanca e Docs", "Roteiro demo", "Falta", "Alta", "", "", "Nao iniciado", "", "Criar pitch <= 10 min com plano B", "docs/DEMO_ROTEIRO.md", ""],
];

tracker.getRange(`A1:K${rows.length + 1}`).values = [headers, ...rows];

tracker.getRange("A1:K1").format = {
  fill: "#1F4E79",
  font: { color: "#FFFFFF", bold: true },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
tracker.getRange(`A2:K${rows.length + 1}`).format = {
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "inside", style: "thin", color: "#D9E2F3" },
};
tracker.getRange(`A1:K${rows.length + 1}`).format.borders = {
  preset: "outside",
  style: "thin",
  color: "#8EA9DB",
};
tracker.getRange("A:A").format.columnWidthPx = 165;
tracker.getRange("B:B").format.columnWidthPx = 190;
tracker.getRange("C:C").format.columnWidthPx = 105;
tracker.getRange("D:D").format.columnWidthPx = 85;
tracker.getRange("E:E").format.columnWidthPx = 145;
tracker.getRange("F:F").format.columnWidthPx = 95;
tracker.getRange("G:G").format.columnWidthPx = 110;
tracker.getRange("H:H").format.columnWidthPx = 90;
tracker.getRange("I:I").format.columnWidthPx = 260;
tracker.getRange("J:J").format.columnWidthPx = 240;
tracker.getRange("K:K").format.columnWidthPx = 240;
tracker.getRange("A1:K1").format.rowHeightPx = 36;
tracker.getRange(`A2:K${rows.length + 1}`).format.rowHeightPx = 44;
tracker.getRange(`H2:H${rows.length + 1}`).format.numberFormat = "0%";

tracker.getRange(`D2:D${rows.length + 1}`).conditionalFormats.add("containsText", {
  text: "Alta",
  format: { fill: "#F8CBAD", font: { color: "#9C0006", bold: true } },
});
tracker.getRange(`G2:G${rows.length + 1}`).conditionalFormats.add("containsText", {
  text: "Concluido",
  format: { fill: "#C6EFCE", font: { color: "#006100", bold: true } },
});
tracker.getRange(`G2:G${rows.length + 1}`).conditionalFormats.add("containsText", {
  text: "Em andamento",
  format: { fill: "#FFEB9C", font: { color: "#9C6500", bold: true } },
});
tracker.getRange(`G2:G${rows.length + 1}`).conditionalFormats.add("containsText", {
  text: "Bloqueado",
  format: { fill: "#FFC7CE", font: { color: "#9C0006", bold: true } },
});

listas.getRange("A1:D1").values = [["Status execucao", "Prioridade", "Responsaveis", "Observacoes"]];
listas.getRange("A2:A7").values = [["Nao iniciado"], ["Em andamento"], ["Em revisao"], ["Bloqueado"], ["Concluido"], ["Cancelado"]];
listas.getRange("B2:B5").values = [["Alta"], ["Media"], ["Baixa"], ["Opcional"]];
listas.getRange("C2:C8").values = [["Voce"], ["Colega Etapa 2"], ["Colega 2"], ["Colega 3"], ["Grupo"], [""], [""]];
listas.getRange("A1:D1").format = {
  fill: "#548235",
  font: { color: "#FFFFFF", bold: true },
  horizontalAlignment: "center",
};
listas.getRange("A:D").format.columnWidthPx = 160;
listas.getRange("A1:D8").format.borders = { preset: "inside", style: "thin", color: "#D9EAD3" };

resumo.getRange("A1:D1").values = [["Resumo do Plano", "", "", ""]];
resumo.getRange("A1:D1").format = {
  fill: "#1F4E79",
  font: { color: "#FFFFFF", bold: true, size: 14 },
};
resumo.getRange("A3:B8").values = [
  ["Total de subetapas", rows.length],
  ["Alta prioridade", ""],
  ["Em andamento", ""],
  ["Bloqueadas", ""],
  ["Concluidas", ""],
  ["Sem responsavel", ""],
];
resumo.getRange("B4:B8").formulas = [
  [`=COUNTIF('Plano de Acao'!D:D,"Alta")`],
  [`=COUNTIF('Plano de Acao'!G:G,"Em andamento")`],
  [`=COUNTIF('Plano de Acao'!G:G,"Bloqueado")`],
  [`=COUNTIF('Plano de Acao'!G:G,"Concluido")`],
  [`=COUNTBLANK('Plano de Acao'!E2:E${rows.length + 1})`],
];
resumo.getRange("A3:B8").format = {
  borders: { preset: "inside", style: "thin", color: "#D9E2F3" },
  wrapText: true,
};
resumo.getRange("A3:A8").format.font = { bold: true };
resumo.getRange("A:A").format.columnWidthPx = 190;
resumo.getRange("B:B").format.columnWidthPx = 120;
resumo.getRange("D3:D9").values = [
  ["Como usar"],
  ["1. Preencha Responsavel, Prazo e Status execucao na aba Plano de Acao."],
  ["2. Use a coluna % concluido para acompanhar progresso."],
  ["3. Atualize Proxima acao e Evidencia/arquivo a cada reuniao."],
  ["4. Evite duas pessoas editarem o mesmo arquivo do projeto ao mesmo tempo."],
  ["5. Priorize itens Alta antes dos itens Media/Baixa."],
  [""],
];
resumo.getRange("D3:D9").format = {
  wrapText: true,
  fill: "#EAF2F8",
  borders: { preset: "outside", style: "thin", color: "#9EADCC" },
};
resumo.getRange("D:D").format.columnWidthPx = 390;

const scan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(scan.ndjson);

await workbook.render({ sheetName: "Plano de Acao", range: "A1:K18", scale: 1 });
await workbook.render({ sheetName: "Resumo", range: "A1:D10", scale: 1 });

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);

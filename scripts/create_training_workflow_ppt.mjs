import pptxgen from "pptxgenjs";
import path from "node:path";

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Grupo Datathon - Precificador Imobiliario";
pptx.company = "Pos Machine Learning Engineering";
pptx.subject = "Passo a passo de treino, DVC, validacao e promocao";
pptx.title = "Pipeline de treino do Precificador Imobiliario";
pptx.lang = "pt-BR";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "pt-BR",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

const C = {
  ink: "18212F",
  muted: "5B667A",
  blue: "2563EB",
  green: "059669",
  amber: "D97706",
  red: "DC2626",
  pale: "F4F7FB",
  line: "D7DEE8",
  codeBg: "101827",
  codeText: "E5E7EB",
  white: "FFFFFF",
};

function addFooter(slide, n) {
  slide.addText(`Precificador Imobiliario | Pipeline ML | ${n}`, {
    x: 0.55,
    y: 7.08,
    w: 12.25,
    h: 0.2,
    fontFace: "Aptos",
    fontSize: 8.5,
    color: "8B95A5",
    margin: 0,
  });
}

function title(slide, text, subtitle) {
  slide.addText(text, {
    x: 0.55,
    y: 0.32,
    w: 9.2,
    h: 0.48,
    fontFace: "Aptos Display",
    fontSize: 27,
    bold: true,
    color: C.ink,
    margin: 0,
    breakLine: false,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.58,
      y: 0.85,
      w: 9.7,
      h: 0.26,
      fontFace: "Aptos",
      fontSize: 11.5,
      color: C.muted,
      margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55,
    y: 1.18,
    w: 1.1,
    h: 0,
    line: { color: C.blue, width: 2.2 },
  });
}

function code(slide, lines, x, y, w, h, fontSize = 9.5) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: C.codeBg },
    line: { color: C.codeBg },
  });
  slide.addText(lines.join("\n"), {
    x: x + 0.18,
    y: y + 0.16,
    w: w - 0.36,
    h: h - 0.28,
    fontFace: "Consolas",
    fontSize,
    color: C.codeText,
    fit: "shrink",
    margin: 0,
    breakLine: false,
  });
}

function pill(slide, label, x, y, color = C.blue) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w: 1.25,
    h: 0.28,
    rectRadius: 0.08,
    fill: { color },
    line: { color },
  });
  slide.addText(label, {
    x,
    y: y + 0.055,
    w: 1.25,
    h: 0.14,
    align: "center",
    fontFace: "Aptos",
    fontSize: 8,
    bold: true,
    color: C.white,
    margin: 0,
  });
}

function bullets(slide, items, x, y, w, opts = {}) {
  const fontSize = opts.fontSize ?? 13;
  const gap = opts.gap ?? 0.38;
  items.forEach((item, i) => {
    slide.addShape(pptx.ShapeType.ellipse, {
      x,
      y: y + i * gap + 0.055,
      w: 0.09,
      h: 0.09,
      fill: { color: opts.color ?? C.blue },
      line: { color: opts.color ?? C.blue },
    });
    slide.addText(item, {
      x: x + 0.2,
      y: y + i * gap,
      w,
      h: 0.25,
      fontFace: "Aptos",
      fontSize,
      color: opts.textColor ?? C.ink,
      margin: 0,
      fit: "shrink",
    });
  });
}

function step(slide, n, label, x, y, color = C.blue) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x,
    y,
    w: 0.45,
    h: 0.45,
    fill: { color },
    line: { color },
  });
  slide.addText(String(n), {
    x,
    y: y + 0.09,
    w: 0.45,
    h: 0.18,
    align: "center",
    fontFace: "Aptos",
    fontSize: 13,
    bold: true,
    color: C.white,
    margin: 0,
  });
  slide.addText(label, {
    x: x + 0.58,
    y: y + 0.08,
    w: 4.7,
    h: 0.22,
    fontFace: "Aptos",
    fontSize: 12.5,
    bold: true,
    color: C.ink,
    margin: 0,
  });
}

function addSlideBg(slide) {
  slide.background = { color: "FFFFFF" };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 7.5,
    fill: { color: "FFFFFF" },
    line: { transparency: 100 },
  });
}

let s = pptx.addSlide();
addSlideBg(s);
s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.ink }, line: { color: C.ink } });
s.addText("Pipeline de treino", {
  x: 0.72,
  y: 0.7,
  w: 7.4,
  h: 0.62,
  fontFace: "Aptos Display",
  fontSize: 34,
  bold: true,
  color: C.white,
  margin: 0,
});
s.addText("Precificador Imobiliario", {
  x: 0.74,
  y: 1.35,
  w: 6,
  h: 0.34,
  fontSize: 17,
  color: "CBD5E1",
  margin: 0,
});
s.addText("DVC, MLflow, validacao, promocao e Git sem subir artefatos pesados", {
  x: 0.74,
  y: 5.95,
  w: 9.4,
  h: 0.34,
  fontSize: 16,
  color: "E2E8F0",
  margin: 0,
});
["features", "train", "validate", "promote", "api"].forEach((label, i) => {
  step(s, i + 1, label, 7.35, 1.05 + i * 0.78, [C.blue, C.green, C.amber, "7C3AED", "0F766E"][i]);
});
s.addText("Versao atual baseada no projeto em execucao", {
  x: 0.74,
  y: 6.42,
  w: 5.7,
  h: 0.24,
  fontSize: 11.5,
  color: "94A3B8",
  margin: 0,
});

s = pptx.addSlide();
addSlideBg(s);
title(s, "1. Preparacao inicial", "Ambiente local e instalacao das dependencias");
bullets(s, [
  "Entrar na pasta raiz do projeto.",
  "Ativar o ambiente virtual .venv.",
  "Em maquina nova, instalar dependencias com pip install -e .",
  "Nunca versionar .venv, data pesada, models ou mlruns."
], 0.75, 1.55, 5.8);
code(s, [
  'cd "C:\\Users\\diesa\\Documents\\Pos - Machine Learning Engineering\\Fase 5\\precificador-imobiliario"',
  "python -m venv .venv",
  ".\\.venv\\Scripts\\activate",
  "pip install -e ."
], 6.4, 1.45, 6.1, 1.65, 9);
code(s, [
  "python -m pytest -q",
  "# esperado no momento: 57 passed"
], 6.4, 3.45, 6.1, 0.92, 9.5);
bullets(s, [
  "Se o ambiente ja existir, rode apenas activate e pytest.",
  "Warnings de bibliotecas sao esperados; erro e teste falhando."
], 6.4, 4.75, 5.7, { color: C.green, fontSize: 12.5 });
addFooter(s, 2);

s = pptx.addSlide();
addSlideBg(s);
title(s, "2. Caminho recomendado: DVC", "Use este caminho quando o DVC estiver funcionando na maquina");
step(s, 1, "features", 0.78, 1.55, C.blue);
step(s, 2, "train", 0.78, 2.22, C.green);
step(s, 3, "validate", 0.78, 2.89, C.amber);
step(s, 4, "repro_check", 0.78, 3.56, "7C3AED");
code(s, [
  "python -m dvc repro features",
  "python -m dvc repro train",
  "python -m dvc repro validate",
  "python -m dvc repro repro_check"
], 5.25, 1.45, 6.85, 1.55, 11);
bullets(s, [
  "O DVC reexecuta apenas o que mudou.",
  "Ele atualiza dvc.lock e os artefatos controlados pelo pipeline.",
  "No fim, confirme que features, treino, validacao e reproducibilidade ficaram OK."
], 5.25, 3.45, 6.25, { fontSize: 12.2 });
code(s, [
  "git add dvc.lock .dvc\\.gitignore",
  "git status --short"
], 5.25, 5.1, 6.85, 0.9, 10.5);
addFooter(s, 3);

s = pptx.addSlide();
addSlideBg(s);
title(s, "3. Saidas esperadas do DVC", "Indicadores que mostram que o pipeline rodou certo");
pill(s, "features", 0.8, 1.45, C.blue);
bullets(s, [
  "data/processed/itbi_features_minimal.csv atualizado.",
  "data/metrics/feature_contract.json com passed = true.",
  "Sem valor_m2 e sem media_valor_cep como features finais."
], 0.8, 1.95, 5.4, { fontSize: 12.2 });
pill(s, "train", 0.8, 3.45, C.green);
bullets(s, [
  "models/dev/model_VERSAO criado.",
  "data/metrics/train_metrics.json atualizado.",
  "Melhor modelo escolhido pelo menor MAE."
], 0.8, 3.95, 5.4, { fontSize: 12.2, color: C.green });
pill(s, "validate", 7.0, 1.45, C.amber);
bullets(s, [
  "data/metrics/validation_dev.json atualizado.",
  "Saida mostra MAE, R2 e faixa de predicao.",
  "A versao carregada precisa ser a mesma gerada no treino."
], 7.0, 1.95, 5.35, { fontSize: 12.2, color: C.amber });
pill(s, "repro", 7.0, 3.45, "7C3AED");
bullets(s, [
  "data/metrics/reproducibility_report.json atualizado.",
  "Relatorio confirma estrutura minima e dependencias essenciais.",
  "O Git deve receber dvc.lock junto com codigo e docs."
], 7.0, 3.95, 5.35, { fontSize: 12.2, color: "7C3AED" });
addFooter(s, 4);

s = pptx.addSlide();
addSlideBg(s);
title(s, "4. Caminho alternativo: comandos diretos", "Use se o DVC der problema de permissao ou cache local");
code(s, [
  "python src\\features\\build_features_minimal.py",
  "python src\\training\\train_mlflow.py",
  "python src\\training\\validate_model.py --env dev",
  "python scripts\\check_reproducibility.py"
], 0.75, 1.55, 7.0, 1.55, 10.5);
bullets(s, [
  "Esse caminho gera os mesmos principais artefatos locais.",
  "Nao atualiza o grafo do DVC do mesmo jeito que dvc repro.",
  "Depois, rode testes e confira metricas antes de promover."
], 0.78, 3.55, 6.8, { fontSize: 12.5 });
code(s, [
  "python -m pytest -q",
  "Get-ChildItem models\\dev",
  "Get-Content data\\metrics\\train_metrics.json"
], 8.0, 1.55, 4.6, 1.32, 9.5);
s.addText("Preferencia do grupo", {
  x: 8.0,
  y: 3.45,
  w: 3.5,
  h: 0.25,
  fontSize: 15,
  bold: true,
  color: C.ink,
  margin: 0,
});
bullets(s, [
  "DVC para execucao oficial.",
  "Comandos diretos para destravar maquina local.",
  "Registrar no README quando usar o caminho alternativo."
], 8.0, 3.9, 4.4, { fontSize: 12.1, color: C.green });
addFooter(s, 5);

s = pptx.addSlide();
addSlideBg(s);
title(s, "5. Promocao por versao", "Promova a versao gerada no treino, nao um modelo ambiguo");
bullets(s, [
  "A versao e a parte depois de model_.",
  "Exemplo: models/dev/model_2026.05.02.1934 -> versao 2026.05.02.1934.",
  "Usar --improvement-pct 0 quando o modelo antigo nao tiver metrica comparavel."
], 0.75, 1.55, 11.5, { fontSize: 12.5 });
code(s, [
  "python src\\training\\promote_model.py --from-env dev --to-env test --version 2026.05.02.1934 --improvement-pct 0",
  "python src\\training\\validate_model.py --env test",
  "",
  "python src\\training\\promote_model.py --from-env test --to-env prod --version 2026.05.02.1934 --improvement-pct 0",
  "python src\\training\\validate_model.py --env prod"
], 0.75, 3.05, 11.9, 2.0, 8.8);
bullets(s, [
  "Se quiser exigir melhora real sobre o ativo, troque 0 por 5.",
  "Se usar 5, o modelo ativo precisa ter metrics.json ou validation.json com mae."
], 0.75, 5.45, 11.4, { fontSize: 12.2, color: C.amber });
addFooter(s, 6);

s = pptx.addSlide();
addSlideBg(s);
title(s, "6. Teste rapido da API", "Depois de promover para prod, valide o contrato real do endpoint");
code(s, [
  "python -m uvicorn api.main:app --reload"
], 0.75, 1.45, 5.7, 0.72, 11);
code(s, [
  'Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `',
  "  -Method Post `",
  '  -ContentType "application/json" `',
  "  -Body '{\"bairro\":\"MOEMA\",\"cep_prefixo\":\"04001\",",
  "          \"area_do_terreno_m2\":120,\"ano\":2024,\"mes\":1}'"
], 0.75, 2.6, 11.85, 1.85, 8.8);
bullets(s, [
  "Se o modelo estiver desalinhado, a API costuma falhar na predicao.",
  "Contrato atual: bairro, cep_prefixo, area_do_terreno_m2, ano, mes.",
  "ano_mes ainda pode ser aceito em algumas camadas, mas ano e mes sao o contrato principal."
], 0.75, 4.85, 11.6, { fontSize: 12.3, color: C.green });
addFooter(s, 7);

s = pptx.addSlide();
addSlideBg(s);
title(s, "7. Git sem subir artefatos pesados", "O .gitignore protege, mas sempre confira antes do commit");
code(s, [
  "git status --short --ignored",
  "git add .gitignore",
  "git add dvc.lock .dvc\\.gitignore",
  "git add api src tests monitoring docs README.md params.yaml pyproject.toml",
  "git status --short",
  'git commit -m "Align ML pipeline with updated feature contract"',
  "git push"
], 0.75, 1.35, 11.85, 2.45, 8.8);
bullets(s, [
  "Nao commitar: .venv, node_modules, data pesada, models, mlruns, caches.",
  "Commitar: codigo, testes, docs, configs, dvc.lock e metricas JSON pequenas.",
  "Se algo pesado entrar staged por engano: git restore --staged caminho/do/arquivo."
], 0.75, 4.25, 11.6, { fontSize: 12.5, color: C.red });
code(s, [
  "python -m dvc push",
  "# somente se houver remoto DVC configurado"
], 0.75, 5.85, 6.8, 0.72, 10);
addFooter(s, 8);

s = pptx.addSlide();
addSlideBg(s);
title(s, "8. Checklist de conclusao", "Sequencia curta para cada pessoa do grupo validar a propria maquina");
const checklist = [
  ["Setup", "Ambiente ativa e pytest passando"],
  ["Features", "feature_contract.json com passed = true"],
  ["Treino", "models/dev/model_VERSAO criado"],
  ["Validacao", "validation_dev.json com MAE e R2"],
  ["Promocao", "test e prod validados com a mesma versao"],
  ["API", "POST /predict respondendo com valor_estimado"],
  ["Git", "status sem artefatos pesados staged"]
];
checklist.forEach((row, i) => {
  const y = 1.45 + i * 0.62;
  slideRow(s, row[0], row[1], y, i);
});
function slideRow(slide, a, b, y, i) {
  slide.addText(a, {
    x: 0.9,
    y,
    w: 2.05,
    h: 0.26,
    fontSize: 12.5,
    bold: true,
    color: [C.blue, C.blue, C.green, C.amber, "7C3AED", "0F766E", C.red][i],
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, { x: 2.75, y: y + 0.14, w: 0.65, h: 0, line: { color: C.line, width: 1.4 } });
  slide.addText(b, {
    x: 3.55,
    y,
    w: 8.2,
    h: 0.26,
    fontSize: 12.5,
    color: C.ink,
    margin: 0,
  });
}
addFooter(s, 9);

s = pptx.addSlide();
addSlideBg(s);
title(s, "9. Erros comuns e como destravar", "Problemas provaveis e acao recomendada");
pill(s, "DVC permissao", 0.75, 1.45, C.red);
bullets(s, [
  "Se dvc repro falhar por acesso local, rode os comandos diretos.",
  "Depois tente DVC novamente ou alinhe com o responsavel pelo ambiente."
], 0.75, 1.95, 5.65, { fontSize: 12.1, color: C.red });
pill(s, "Promocao", 0.75, 3.1, C.amber);
bullets(s, [
  "Se o ativo antigo nao tem mae, usar --improvement-pct 0.",
  "Se usar 5, garanta metrics.json/validation.json no modelo ativo."
], 0.75, 3.6, 5.65, { fontSize: 12.1, color: C.amber });
pill(s, "API", 7.0, 1.45, C.blue);
bullets(s, [
  "Se /predict falhar, confirmar que prod usa modelo novo.",
  "Payload deve conter bairro, cep_prefixo, area, ano e mes."
], 7.0, 1.95, 5.4, { fontSize: 12.1 });
pill(s, "Git", 7.0, 3.1, C.green);
bullets(s, [
  "Antes do commit, revisar git status --short.",
  "Nao subir data pesada, models, mlruns, .venv ou node_modules."
], 7.0, 3.6, 5.4, { fontSize: 12.1, color: C.green });
addFooter(s, 10);

s = pptx.addSlide();
addSlideBg(s);
title(s, "10. Divisao sugerida do grupo", "Mantem trabalho em paralelo mesmo com uma branch principal");
bullets(s, [
  "Pessoa 1: dados, DVC, feature_contract e dvc.lock.",
  "Pessoa 2: treino, MLflow, metricas e promocao de modelos.",
  "Pessoa 3: API, agent, guardrails e testes de endpoint.",
  "Pessoa 4: documentacao, checklist do PDF, benchmark e README.",
  "Antes de cada push: git pull, rodar pytest, revisar git status."
], 0.75, 1.55, 11.5, { fontSize: 14, gap: 0.56 });
code(s, [
  "git pull",
  "python -m pytest -q",
  "git status --short",
  "git add ...",
  "git commit -m \"mensagem clara\"",
  "git push"
], 0.75, 5.0, 11.85, 1.28, 10);
addFooter(s, 11);

const out = path.resolve("docs", "passo_a_passo_treino_dvc_precificador.pptx");
await pptx.writeFile({ fileName: out });
console.log(out);

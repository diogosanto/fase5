# Prometheus Quickstart README

Um guia passo a passo para configurar Prometheus localmente, adicionar um target de API e criar uma regra de alerta simples.

---

### Pré requisitos
- **Sistema**: Windows com PowerShell.  
- **Binaries**: `prometheus.exe` disponível em `C:\prometheus3112`.  
- **Ports**: Prometheus em **9090**, API em **127.0.0.1:8000**.  
- **Editor**: Notepad ou VS Code (salvar arquivos em **UTF-8**).

---

### Estrutura de diretório e arquivos essenciais
Coloque os arquivos abaixo em `C:\prometheus3112`:

C:\prometheus3112
├─ prometheus.exe
├─ prometheus.yml
└─ alertrules.yml


**prometheus.yml** (exemplo mínimo)
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api_precificador'
    static_configs:
      - targets: ['127.0.0.1:8000']

rule_files:
  - "alertrules.yml"




**alertrules.yml** (exemplo mínimo)
groups:
  - name: api_alerts
    rules:
      - alert: ApiDown
        expr: up{job="api_precificador"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API inacessível"
          description: "O target api_precificador está com up == 0."


Observações
Salve os arquivos em UTF-8.
Use extensão .yml ou .yaml.
Remova arquivos com nomes parecidos (prometheus.txt, prometheus.fnote) para evitar confusão.


Comandos para iniciar, recarregar e validar
Iniciar Prometheus

powershell
cd C:\prometheus3112
.\prometheus.exe --config.file=C:\prometheus3112\prometheus.yml --web.listen-address=":9090" --storage.tsdb.path="data" --web.enable-lifecycle
Recarregar configuração sem reiniciar

powershell
Invoke-WebRequest -Uri http://localhost:9090/-/reload -Method POST
Verificar targets e health

powershell
Invoke-RestMethod -Uri http://localhost:9090/api/v1/targets | ConvertTo-Json -Depth 10
Verificar valor de up

powershell
Invoke-RestMethod -Uri 'http://localhost:9090/api/v1/query?query=up{job="api_precificador"}' | ConvertTo-Json -Depth 6
Verificar regras carregadas

powershell
Invoke-RestMethod -Uri http://localhost:9090/api/v1/rules | ConvertTo-Json -Depth 6
Verificar alerts

powershell
Invoke-RestMethod -Uri http://localhost:9090/api/v1/alerts | ConvertTo-Json -Depth 6
Teste do alerta ApiDown passo a passo


Checklist rápido para replicação
[ ] Colocar prometheus.exe, prometheus.yml, alertrules.yml em C:\prometheus3112.

[ ] Iniciar Prometheus com --web.enable-lifecycle.

[ ] Confirmar http://localhost:9090/targets e up{job="api_precificador"} == 1.

[ ] Salvar alertrules.yml em UTF-8 e recarregar via /-/reload.

[ ] Testar ApiDown parando a API por 1 minuto e verificar alerts.

[ ] Versionar configs e configurar Alertmanager.
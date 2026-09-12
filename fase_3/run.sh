# Classificações únicas
for i in {1..20}; do
  curl -s -X POST "http://localhost:8000/v1/classify" \
    -H "Content-Type: application/json" \
    -d '{"text": "Patient with medical condition requiring diagnosis"}' > /dev/null
done

# Classificações em lote
curl -s -X POST "http://localhost:8000/v1/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Patient with malignant neoplasm",
      "Gastroesophageal reflux disease",
      "MRI showing multiple sclerosis"
    ]
  }'

# Gerar erros (para testar painel de taxa de erro)
for i in {1..10}; do
  curl -s -X POST "http://localhost:8000/v1/classify" \
    -H "Content-Type: application/json" \
    -d '{"text": ""}' > /dev/null
done
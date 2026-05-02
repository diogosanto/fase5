def validate_input(data):
    bairro = str(data.get("bairro", "")).strip()
    if len(bairro) < 2:
        raise ValueError("Bairro invalido")

    if float(data["area_do_terreno_m2"]) <= 0:
        raise ValueError("Area invalida")

    if "cep_prefixo" in data and not str(data["cep_prefixo"]).strip():
        raise ValueError("CEP prefixo invalido")

    if "mes" in data and not 1 <= int(data["mes"]) <= 12:
        raise ValueError("Mes invalido")


def validate_output(value):
    if value < 0:
        raise ValueError("Saida invalida do modelo")

    return value

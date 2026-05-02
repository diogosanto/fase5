def validate_input(data):
    if data["area_do_terreno_m2"] <= 0:
        raise ValueError("Área inválida")

    if data["valor_m2"] <= 0:
        raise ValueError("Valor inválido")

    if len(data["bairro"]) < 2:
        raise ValueError("Bairro inválido")


def validate_output(value):
    if value < 0:
        raise ValueError("Saída inválida do modelo")

    return value
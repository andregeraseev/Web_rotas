# views.py
from django.shortcuts import render
from .forms import EnderecoFormSet, TecnicoFormSet, ParametrosForm
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def routing_view(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            data = json.loads(request.body)

            endereco_formset = data.get('endereco_formset', [])
            tecnico_formset = data.get('tecnico_formset', [])
            numero_tecnico = int(data.get('num_tecnicos', 0))
            max_stops = int(data.get('max_stops_input', 0))
            max_time = int(data.get('max_time_input', 0))
            print('Endereços:')
            for endereco in endereco_formset:
                print(endereco)

            print('Técnicos: ', numero_tecnico)
            print('Paradas: ', max_stops)
            print('Tempo: ', max_time)
            partida = []
            print('tecnico_formset', tecnico_formset)
            # Inicializa as listas de partida e volta
            partida = []
            volta = []

            # Itera sobre cada técnico no tecnico_formset
            for tecnico in tecnico_formset:
                # Adiciona a partida e a volta às respectivas listas
                partida.append(tecnico.get('partida'))
                volta.append(tecnico.get('final'))

            # Após o loop, você terá as listas completas de partida e volta
            print('Partida:', partida)
            print('Volta:', volta)


            solution, rotas, total_distance, planos_rotas = solve_routing_problem(endereco_formset, partida, volta, numero_tecnico,
                                      max_stops, max_time, max_distance_per_vehicle=1000000)
            if solution:
            # Adiciona um botão para cada rota ao layout
                for i, rota in enumerate(rotas):
                    print('rota', i, rota)

                print('rotas', rotas)
                return JsonResponse({'status': 'success', 'rotas': rotas, 'total_distance': total_distance, 'planos_rotas': planos_rotas})

            else:
                return JsonResponse({'status': 'error', 'message': 'Nenhuma solução encontrada!'})

        else:
            endereco_formset = EnderecoFormSet(request.POST, prefix='enderecos')
            tecnico_formset = TecnicoFormSet(request.POST, prefix='tecnicos')
            parametros_formset = ParametrosForm(prefix='parametros')

            if endereco_formset.is_valid() and tecnico_formset.is_valid():
                # Processa os Formsets...
                for form in endereco_formset:
                    endereco = form.cleaned_data.get('endereco')
                    print(endereco)

                for form in tecnico_formset:
                    partida = form.cleaned_data.get('partida')
                    final = form.cleaned_data.get('final')
                    print('partida',partida, 'final',final)
    else:
        endereco_formset = EnderecoFormSet(prefix='enderecos')
        tecnico_formset = TecnicoFormSet(prefix='tecnicos')
        parametros_formset = ParametrosForm(prefix='parametros')

    return render(request, 'index.html', {
        'parametros_formset': parametros_formset,
        'endereco_formset': endereco_formset,
        'tecnico_formset': tecnico_formset,
    })









import googlemaps
from datetime import datetime
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from urllib.parse import quote
# Sua chave da API do Google Maps
google_maps_key='AIzaSyAgAeYRe2m-qz-UU-xFVvNCWF_uVe-Kaeo'
import numpy as np
def calculate_distances_matrix(addresses, google_maps_key):
    batch_size = 10  # Tamanho do lote para chamadas de API
    gmaps = googlemaps.Client(key=google_maps_key)
    n = len(addresses)

    # Inicializar matrizes completas com zeros
    full_distance_matrix = np.zeros((n, n), dtype=int)
    full_time_matrix = np.zeros((n, n), dtype=int)

    for i in range(0, n, batch_size):
        for j in range(0, n, batch_size):
            # Define os lotes de origens e destinos
            origins_batch = addresses[i:i + batch_size]
            destinations_batch = addresses[j:j + batch_size]

            # Filtra endereços nulos
            origins_batch = [addr for addr in origins_batch if addr is not None]
            destinations_batch = [addr for addr in destinations_batch if addr is not None]

            if not origins_batch or not destinations_batch:
                continue  # Pula iteração se o lote estiver vazio

            # Faz a chamada para a API
            matrix_result = gmaps.distance_matrix(origins=origins_batch,
                                                  destinations=destinations_batch,
                                                  mode="driving",
                                                  departure_time=datetime.now())

            # Processa os resultados e preenche as matrizes completas
            for oi, row in enumerate(matrix_result['rows']):
                for di, element in enumerate(row['elements']):
                    if 'distance' in element and 'duration' in element:
                        distance = element['distance']['value']
                        duration = element['duration']['value']
                        full_distance_matrix[i + oi][j + di] = distance
                        full_time_matrix[i + oi][j + di] = duration

    return full_distance_matrix.tolist(), full_time_matrix.tolist()

def create_data_model(enderecos, partida, volta, num_tecnicos):
    print('create_data_model' )
    print('partida', partida, volta,)
    data = {}
    distance, time_matrix = calculate_distances_matrix(
        enderecos + [addr for addr in partida] + [addr for addr in volta], google_maps_key)
    data['distance_matrix'] = distance
    data['time_matrix'] = time_matrix
    data['num_vehicles'] = num_tecnicos

    # Ajusta os índices de partida e volta somente para os endereços não nulos
    data['starts'] = [len(enderecos) + i for i, addr in enumerate(partida)]
    data['ends'] = [len(enderecos) + len(partida) + i for i, addr in enumerate(volta)]

    return data

def create_balance_dimension(routing, manager, max_visits, data):
    """Adiciona uma dimensão para balancear o número de visitas entre os veículos."""

    def count_callback(from_index):
        """Retorna 1 para cada visita, exceto para os pontos de partida/chegada."""
        from_node = manager.IndexToNode(from_index)
        if from_node in data['starts'] or from_node in data['ends']:
            return 0  # Não conta como visita
        return 1  # Conta como uma visita

    # Registrar a função de callback no modelo de roteamento.
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)

    # Adiciona a dimensão de balanceamento ao modelo.
    routing.AddDimension(
        count_callback_index,
        0,  # slack max: não aplicável aqui, pois não estamos permitindo folga
        max_visits,  # capacidade máxima: o valor máximo que a dimensão pode ter
        True,  # start cumul to zero: inicia a acumulação em zero
        'Balance'
    )


def solve_routing_problem(enderecos_list, partida, volta, num_tecnicos,
                                                          max_stops_per_vehicle, max_time_per_vehicle,max_distance_per_vehicle):
    print('solve_routing_problem')
#     # Lista de endereços

    num_tecnicos = num_tecnicos
    enderecos = enderecos_list
    partida = partida
    volta = volta


    # Cria o modelo de dados.
    max_time_per_vehicle = max_time_per_vehicle*60  # Tempo máximo total por veículo
    data = create_data_model(enderecos, partida, volta, num_tecnicos)
    print('data',data)

    print("Inicializando o RoutingIndexManager...")
    # Cria o gerenciador de roteamento e o modelo de roteamento.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['starts'], data['ends'])
    print("RoutingIndexManager inicializado com sucesso.")

    print("Inicializando o routing...")
    routing = pywrapcp.RoutingModel(manager)
    print("routing inicializado com sucesso.")

    print("Inicializando o create_balance_dimension...")
    create_balance_dimension(routing, manager, max_stops_per_vehicle, data)
    print("Finalizando o create_balance_dimension...")

    max_stops_per_vehicle = max_stops_per_vehicle  # Número mínimo de paradas por técnico


    # create_minimum_stops_dimension(routing, manager, data, min_stops_per_vehicle)

    PENALIDADE_POR_PARADA = 1800  # 10 unidades de tempo de penalidade por parada

    def time_callback(from_index, to_index):
        # Convertendo índices de roteamento para índices da matriz
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Adicionando penalidade por parada
        # print('time_callback', from_node, to_node)
        if from_node != to_node:  # Evitar penalidade para "paradas" no mesmo local
            return data['time_matrix'][from_node][to_node] + PENALIDADE_POR_PARADA
        else:
            return data['time_matrix'][from_node][to_node]

    print("Iniciando o transit_callback_index...")
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    print("Finalizando o transit_callback_index...")
    print("Iniciando o routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)...")
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    print("Finalizando o routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)...")
    # print('transit_callback_index',transit_callback_index)
    routing.AddDimension(
        transit_callback_index,
        5,  # buffer ou tempo de espera permitido entre as entregas
        max_time_per_vehicle,  # máximo de tempo total por rota
        True,  # Não force o início cumulativo a zero
        'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    time_dimension.SetGlobalSpanCostCoefficient(100)



    # Cria e registra uma função de custo de trânsito.
    def distance_callback(from_index, to_index):
      # Convertendo os índices para os nós correspondentes
      from_node = manager.IndexToNode(from_index)
      to_node = manager.IndexToNode(to_index)
      # print(' distance_callback from_node', from_node)
      # Retornando a distância entre os nós
      return data['distance_matrix'][from_node][to_node]

    print("Inicializando o transit_callback_index...")
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    print("Finalizando o transit_callback_index...")
    print("Inicializando o routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)...")
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    print("Finalizando o routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)...")
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        max_distance_per_vehicle,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(10)




    # Configuração de pesquisa para a solução.
    print("Inicializando o search_parameters...")
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    print("Finalizando o search_parameters...")
    print("Inicializando o search_parametersfirst_solution_strategy...")
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    print("Inicializando o search_parametersfirst_solution_strategy...")
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    print("Finalizando o search_parametersfirst_solution_strategy...")
    search_parameters.time_limit.seconds = 10  # Defina um limite de tempo para a pesquisa, se necessário

    # Soluciona o problema.
    print("Inicializando o SolveWithParameters...")
    solution = routing.SolveWithParameters(search_parameters)
    print("Finalizando o SolveWithParameters...")

    # Imprime a solução.
    if solution:
        balance_dimension = routing.GetDimensionOrDie('Balance')
        for vehicle_id in range(data['num_vehicles']):
            index = routing.End(vehicle_id)
            carga_final = solution.Value(balance_dimension.CumulVar(index))
            print(f"Carga final para veículo {vehicle_id}: {carga_final} visitas.")
        print('Solução encontrada!\n')
        total_distance = 0
        rotas = []
        planos_rotas = []
        for vehicle_id in range(data['num_vehicles']):
            # print(f'Rota do Técnico {vehicle_id}:')
            index = routing.Start(vehicle_id)
            plan_output = f''
            route_link = "https://www.google.com/maps/dir/"
            route_distance = 0

            while not routing.IsEnd(index):
                # print('index', index)
                node_index = manager.IndexToNode(index)

                if node_index < len(enderecos):
                    # print('node_index', node_index)
                    plan_output += f' {enderecos[node_index]} -> '
                    encoded_address = quote(enderecos[node_index])
                else:
                    # print('node_index', node_index)
                    extra_index = node_index - len(enderecos)

                    if extra_index < len(partida):
                        if partida[extra_index] is not None:
                            # print('extra_index <', extra_index)
                            # print('partida[extra_index]', partida[extra_index])
                            plan_output += f' {partida[extra_index]} -> '
                            encoded_address = quote(partida[extra_index])
                        else:
                            # Pular para o próximo ponto se partida for None
                            previous_index = index
                            index = solution.Value(routing.NextVar(index))
                            continue
                    elif extra_index - len(partida) < len(volta):
                        extra_index -= len(partida)
                        if volta[extra_index] is not None:
                            # print('extra_index', extra_index)
                            plan_output += f' {volta[extra_index]} -> '
                            # print('volta[extra_index]', volta[extra_index])
                            encoded_address = quote(volta[extra_index])
                            # print('encoded_address', encoded_address)
                        else:
                            # Pular para o próximo ponto se volta for None
                            previous_index = index
                            index = solution.Value(routing.NextVar(index))
                            continue
                route_link += encoded_address.replace(' ', '+') + '/'
                previous_index = index
                index = solution.Value(routing.NextVar(index))  # Atualiza o índice para o próximo ponto
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            end_node_index = manager.IndexToNode(routing.End(vehicle_id))
            if end_node_index < len(enderecos):
                plan_output += f'{enderecos[end_node_index]}\n'
                encoded_address = quote(enderecos[end_node_index])
            else:
                extra_index = end_node_index - len(enderecos)
                if extra_index < len(partida):
                    if partida[extra_index] is not None:
                        plan_output += f'{partida[extra_index]}\n'
                        encoded_address = quote(partida[extra_index])
                    else:
                        # Se partida for None, não adicionar ao plano de rota
                        encoded_address = ''
                elif extra_index - len(partida) < len(volta):
                    extra_index -= len(partida)
                    if volta[extra_index] is not None:
                        plan_output += f'{volta[extra_index]}\n'
                        encoded_address = quote(volta[extra_index])
                    else:
                        # Se volta for None, não adicionar ao plano de rota
                        encoded_address = ''

            if encoded_address:
                route_link += encoded_address.replace(' ', '+')
            plan_output += f'Distância da Rota: {route_distance}m\n'
            print(plan_output)
            planos_rotas.append(plan_output)
            print("Link do Google Maps para esta rota:", route_link)
            rotas.append(route_link)
            total_distance += route_distance
        print(f'Distância Total: {total_distance}m')
        print('lista de rotas kivy', rotas)
        return solution, rotas, total_distance, planos_rotas
    else:
      print('Nenhuma solução encontrada!')
      return None, None, None, None

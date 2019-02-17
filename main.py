import numpy as np
import asyncio as aio
import websockets as ws
import neuro as bn
import json

nets = []
nets = bn.multiple_instances([3, 5, 7, 3], 10)


async def ann_server(websocket, path):
    data = await websocket.recv()

    response = ''
    if data.startswith('solve'):
        response = dispatch_solve(data)

    await websocket.send(response)


def dispatch_solve(data):
    data = data.replace('solve', '')
    data_object = json.loads(data)

    inputs = extract_inputs(data_object)
    outputs = proccess_inputs(inputs)
    response_object = pack_outputs(outputs)

    response = json.dumps(response_object)
    return response


def extract_inputs(data_object):
    inputs = []
    for p in data_object['vectors']:
        vec3 = p['vector']
        inputs.append(np.matrix([[vec3['x'], vec3['y'], vec3['z']]]).T)

    return inputs


def proccess_inputs(inputs):
    inputs_count = len(inputs)
    outputs = []

    if len(inputs) == len(nets):

        for i in range(0, inputs_count):
            outputs.append(nets[i].run(inputs[i]))

    return outputs


def pack_outputs(outputs):
    throw_vecs = []
    for o in outputs:
        vec = {'x': o[0, 0], 'y': o[1, 0], 'z': o[2, 0]}
        throw_vecs.append({'vector': vec})

    return {'vectors': throw_vecs}


def main():
    start_server = ws.serve(ann_server, 'localhost', 8765)
    aio.get_event_loop().run_until_complete(start_server)
    aio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()

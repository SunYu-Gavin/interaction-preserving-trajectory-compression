"""Inspect pickle opcodes as inert data; never import globals or invoke reducers.

This is a deliberately limited metadata syntax reader, not an unpickler.
Unsupported opcodes fail closed. Constructor/reducer names remain plain strings.
"""
from dataclasses import dataclass, field
import pickletools


@dataclass
class Node:
    symbol: str
    args: object = ()
    state: object = field(default_factory=dict)
    items: dict = field(default_factory=dict)

    def __setitem__(self, key, value):
        self.items[key] = value


def inspect_metadata(blob: bytes):
    if len(blob) > 16 * 1024 * 1024:
        raise ValueError("Metadata exceeds static inspection budget")
    stack, memo = [], {}
    mark = object()

    def group():
        i = len(stack)-1
        while i >= 0 and stack[i] is not mark:
            i -= 1
        if i < 0:
            raise ValueError("Missing metadata stack marker")
        values = stack[i+1:]
        del stack[i:]
        return values

    for count, (op, arg, pos) in enumerate(pickletools.genops(blob)):
        if count > 1000000:
            raise ValueError("Metadata opcode budget exceeded")
        name = op.name
        if name in {"PROTO", "FRAME"}:
            continue
        if name in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "SHORT_BINBYTES", "BINBYTES", "BININT", "BININT1", "BININT2", "BINFLOAT", "LONG1", "LONG4"}:
            stack.append(arg)
        elif name in {"NONE", "NEWTRUE", "NEWFALSE"}:
            stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
        elif name == "MARK":
            stack.append(mark)
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in {"BINPUT", "LONG_BINPUT"}:
            memo[arg] = stack[-1]
        elif name in {"BINGET", "LONG_BINGET"}:
            stack.append(memo[arg])
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name in {"EMPTY_LIST", "EMPTY_SET"}:
            stack.append([])
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "TUPLE":
            stack.append(tuple(group()))
        elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
            n = int(name[-1]); values = tuple(stack[-n:]); del stack[-n:]; stack.append(values)
        elif name in {"APPENDS", "ADDITEMS"}:
            values = group(); stack[-1].extend(values)
        elif name == "APPEND":
            value = stack.pop(); stack[-1].append(value)
        elif name == "SETITEMS":
            values = group()
            if len(values) % 2:
                raise ValueError("Unpaired metadata dictionary entries")
            for i in range(0, len(values), 2):
                stack[-1][values[i]] = values[i+1]
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop(); stack[-1][key] = value
        elif name == "STACK_GLOBAL":
            symbol, module = stack.pop(), stack.pop()
            if not isinstance(symbol, str) or not isinstance(module, str):
                raise ValueError("Invalid metadata symbol")
            stack.append(Node(module+"."+symbol))
        elif name in {"REDUCE", "NEWOBJ"}:
            args, symbol = stack.pop(), stack.pop()
            if not isinstance(symbol, Node):
                raise ValueError("Invalid inert constructor")
            stack.append(Node(symbol.symbol, args))
        elif name == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], Node):
                raise ValueError("Invalid inert object state")
            stack[-1].state = state
        elif name == "STOP":
            if len(stack) != 1 or pos != len(blob)-1:
                raise ValueError("Unexpected metadata tail/stack")
            return stack[0]
        else:
            raise ValueError(f"Unsupported static metadata opcode: {name}")
    raise ValueError("Missing metadata STOP")


def vehicle_metadata(blob: bytes):
    scene = inspect_metadata(blob)
    if not isinstance(scene, Node) or scene.symbol != "trajdata.data_structures.scene_metadata.Scene":
        raise ValueError("Expected trajdata scene metadata")
    agents = scene.state.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Missing complete agent metadata")
    types = {}
    for agent in agents:
        if not isinstance(agent, Node) or agent.symbol != "trajdata.data_structures.agent.AgentMetadata":
            raise ValueError("Unexpected agent metadata class")
        state = agent.state; typ = state.get("type")
        if not isinstance(typ, Node) or typ.symbol != "trajdata.data_structures.agent.AgentType" or len(typ.args) != 1:
            raise ValueError("Unrecognized agent type")
        name = state["name"]
        if not isinstance(name, str) or name in types or typ.args[0] not in range(5):
            raise ValueError("Invalid or duplicate agent identity/type")
        types[name] = typ.args[0]
    return scene.state, types

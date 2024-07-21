// Based on https://github.com/socketio/socket.io-msgpack-parser/blob/main/index.js
import { pack, unpack } from 'msgpackr'
import Emitter from 'component-emitter'

export const protocol = 5;

/**
 * Packet types (see https://github.com/socketio/socket.io-protocol)
 */

export const PacketType = {
  CONNECT: 0,
  DISCONNECT: 1,
  EVENT: 2,
  ACK: 3,
  CONNECT_ERROR: 4,
};

const isInteger =
  Number.isInteger ||
  function (value) {
    return (
      typeof value === "number" &&
      isFinite(value) &&
      Math.floor(value) === value
    );
  };

const isString = function (value: any) {
  return typeof value === "string";
};

const isObject = function (value: any) {
  return Object.prototype.toString.call(value) === "[object Object]";
};

export function Encoder() {}

Encoder.prototype.encode = function (packet: any) {
  return [pack(packet)];
};

export function Decoder() {}

Emitter(Decoder.prototype);

Decoder.prototype.add = function (obj: any) {
  const decoded = unpack(obj);
  this.checkPacket(decoded);
  this.emit("decoded", decoded);
};

function isDataValid(decoded: any) {
  switch (decoded.type) {
    case PacketType.CONNECT:
      return decoded.data === undefined || isObject(decoded.data);
    case PacketType.DISCONNECT:
      return decoded.data === undefined;
    case PacketType.CONNECT_ERROR:
      return isString(decoded.data) || isObject(decoded.data);
    default:
      return Array.isArray(decoded.data);
  }
}

Decoder.prototype.checkPacket = function (decoded: any) {
  const isTypeValid =
    isInteger(decoded.type) &&
    decoded.type >= PacketType.CONNECT &&
    decoded.type <= PacketType.CONNECT_ERROR;
  if (!isTypeValid) {
    throw new Error("invalid packet type");
  }

  if (!isString(decoded.nsp)) {
    throw new Error("invalid namespace");
  }

  if (!isDataValid(decoded)) {
    throw new Error("invalid payload");
  }

  const isAckValid = decoded.id === undefined || isInteger(decoded.id);
  if (!isAckValid) {
    throw new Error("invalid packet id");
  }
};

Decoder.prototype.destroy = function () {};

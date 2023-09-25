export class BlockingQueue<T> {
    private _promises: Promise<T>[];
    private _resolvers: ((t: T) => void)[];

    constructor() {
        this._resolvers = [];
        this._promises = [];
    }

    private _add() {
        this._promises.push(
            new Promise((resolve) => {
                this._resolvers.push(resolve);
            })
        );
    }

    enqueue(t: T) {
        if (this._resolvers.length == 0) this._add();
        const resolve = this._resolvers.shift()!;
        resolve(t);
    }

    dequeue() {
        if (this._promises.length == 0) this._add();
        const promise = this._promises.shift()!;
        return promise;
    }

    isEmpty() {
        return this._promises.length == 0;
    }

    isBlocked() {
        return this._resolvers.length != 0;
    }

    get length() {
        return this._promises.length - this._resolvers.length;
    }
}

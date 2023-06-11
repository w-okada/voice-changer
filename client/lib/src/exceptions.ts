export class ModelLoadException extends Error {
    public causeFileType: string = ""
    constructor(causeFileType: string) {
        super(`Model Load Exception:${causeFileType}`);
        this.causeFileType = causeFileType;
        this.name = this.constructor.name;
        Error.captureStackTrace(this);
    }
}

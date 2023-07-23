import React, { ErrorInfo } from "react";

type ErrorBoundaryProps = {
    children: React.ReactNode;
    fallback: React.ReactNode;
    onError: (error: Error, errorInfo: React.ErrorInfo | null, reason: any) => void;
};

type ErrorBoundaryState = {
    hasError: boolean;
};

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
    private eventHandler: () => void;
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = { hasError: false };
        this.eventHandler = this.updateError.bind(this);
    }

    static getDerivedStateFromError(_error: Error) {
        // console.warn("React Error Boundary Catch", error)
        return { hasError: true };
    }
    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        // For logging
        console.warn("React Error Boundary Catch", error, errorInfo);
        const { onError } = this.props;
        if (onError) {
            onError(error, errorInfo, null);
        }
    }

    // 非同期例外対応
    updateError() {
        this.setState({ hasError: true });
    }
    handledRejection = (event: PromiseRejectionEvent) => {
        const { onError } = this.props;
        const error = new Error(event.type);
        onError(error, null, event.reason);
        this.setState({ hasError: true });
    };
    componentDidMount() {
        // window.addEventListener('unhandledrejection', this.eventHandler)
        window.addEventListener("unhandledrejection", this.handledRejection);
    }

    componentWillUnmount() {
        // window.removeEventListener('unhandledrejection', this.eventHandler)
        window.removeEventListener("unhandledrejection", this.handledRejection);
    }

    render() {
        if (this.state.hasError) {
            return this.props.fallback;
        }
        return this.props.children;
    }
}

export default ErrorBoundary;

import React, { createRef, ErrorInfo } from 'react';

type ErrorBoundaryProps = {
    children: React.ReactNode;
    fallback: React.ReactNode;
    onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

type ErrorBoundaryState = {
    hasError: boolean;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
    private eventHandler: () => void
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
        console.warn("React Error Boundary Catch", error, errorInfo)
        const { onError } = this.props;
        if (onError) {
            onError(error, errorInfo);
        }
    }


    // 非同期例外対応
    updateError() {
        this.setState({ hasError: true });
    }
    componentDidMount() {
        window.addEventListener('unhandledrejection', this.eventHandler)
    }

    componentWillUnmount() {
        window.removeEventListener('unhandledrejection', this.eventHandler)
    }

    render() {
        if (this.state.hasError) {
            return this.props.fallback;
        }
        return this.props.children;
    }
}


export default ErrorBoundary;
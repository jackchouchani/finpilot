import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, Typography, CircularProgress } from '@mui/material';

function HistoricalDataAnalysis() {
    const [ticker, setTicker] = useState('');
    const [analysis, setAnalysis] = useState('');
    const [loading, setLoading] = useState(false);

    const analyzeHistoricalData = async () => {
        setLoading(true);
        try {
            const response = await axios.post('/previous_day_analysis', { ticker });
            setAnalysis(response.data.analysis);
        } catch (error) {
            console.error('Error analyzing historical data:', error);
            setAnalysis('Error analyzing data');
        }
        setLoading(false);
    };

    return (
        <div>
            <Typography variant="h6">Previous Day Data Analysis</Typography>
            <TextField
                label="Stock Ticker"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
            />
            <Button onClick={analyzeHistoricalData} disabled={loading}>
                Analyze Data
            </Button>
            {loading ? (
                <CircularProgress />
            ) : (
                <Typography>{analysis}</Typography>
            )}
        </div>
    );
}

export default HistoricalDataAnalysis;
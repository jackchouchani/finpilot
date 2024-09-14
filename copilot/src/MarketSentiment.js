import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, Typography, CircularProgress } from '@mui/material';

function MarketSentiment() {
    const [ticker, setTicker] = useState('');
    const [sentiment, setSentiment] = useState('');
    const [loading, setLoading] = useState(false);

    const analyzeSentiment = async () => {
        setLoading(true);
        try {
            const response = await axios.post('/market_sentiment', { ticker });
            setSentiment(response.data.sentiment);
        } catch (error) {
            console.error('Error analyzing market sentiment:', error);
            setSentiment('Error analyzing sentiment');
        }
        setLoading(false);
    };

    return (
        <div>
            <Typography variant="h6">Market Sentiment Analysis</Typography>
            <TextField
                label="Stock Ticker"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
            />
            <Button onClick={analyzeSentiment} disabled={loading}>
                Analyze Sentiment
            </Button>
            {loading ? (
                <CircularProgress />
            ) : (
                <Typography>{sentiment}</Typography>
            )}
        </div>
    );
}

export default MarketSentiment;
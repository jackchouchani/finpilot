import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, Typography, CircularProgress, Select, MenuItem } from '@mui/material';

function InvestmentRecommendation() {
    const [portfolio, setPortfolio] = useState('');
    const [riskProfile, setRiskProfile] = useState('moderate');
    const [recommendation, setRecommendation] = useState('');
    const [loading, setLoading] = useState(false);

    const getRecommendation = async () => {
        setLoading(true);
        try {
            const response = await axios.post('/investment_recommendation', {
                portfolio: portfolio.split(',').map(stock => stock.trim()),
                risk_profile: riskProfile
            });
            setRecommendation(response.data.recommendation);
        } catch (error) {
            console.error('Error getting investment recommendation:', error);
            setRecommendation('Error getting recommendation');
        }
        setLoading(false);
    };

    return (
        <div>
            <Typography variant="h6">Investment Recommendation</Typography>
            <TextField
                label="Portfolio (comma-separated tickers)"
                value={portfolio}
                onChange={(e) => setPortfolio(e.target.value)}
            />
            <Select
                value={riskProfile}
                onChange={(e) => setRiskProfile(e.target.value)}
            >
                <MenuItem value="conservative">Conservative</MenuItem>
                <MenuItem value="moderate">Moderate</MenuItem>
                <MenuItem value="aggressive">Aggressive</MenuItem>
            </Select>
            <Button onClick={getRecommendation} disabled={loading}>
                Get Recommendation
            </Button>
            {loading ? (
                <CircularProgress />
            ) : (
                <Typography>{recommendation}</Typography>
            )}
        </div>
    );
}

export default InvestmentRecommendation;
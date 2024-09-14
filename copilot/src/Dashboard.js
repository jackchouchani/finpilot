import React from 'react';
import { Grid, Paper, Typography } from '@mui/material';
import { PortfolioPieChart } from './Portfolio';  // Assurez-vous d'exporter PortfolioPieChart depuis Portfolio.js

function Dashboard({ portfolio }) {
    console.log('Dashboard portfolio:', portfolio); // Pour déboguer

    if (!portfolio) {
        return <Typography>Chargement du portfolio...</Typography>;
    }

    if (!portfolio.stocks || portfolio.stocks.length === 0) {
        return <Typography>Aucune donnée de portfolio disponible</Typography>;
    }

    // const totalValue = portfolio.stocks.reduce((sum, stock) => sum + parseFloat(stock.weight) * parseFloat(stock.entryPrice), 0);

    // return (
    //     <Grid container spacing={3}>
    //         <Grid item xs={12} md={6}>
    //             <Paper elevation={3} style={{ padding: '20px' }}>
    //                 <Typography variant="h6">Valeur du Portfolio</Typography>
    //                 <Typography variant="h4">${totalValue.toFixed(2)}</Typography>
    //             </Paper>
    //         </Grid>
    //         <Grid item xs={12} md={6}>
    //             <Paper elevation={3} style={{ padding: '20px' }}>
    //                 <Typography variant="h6">Allocation du Portfolio</Typography>
    //                 <PortfolioPieChart portfolio={portfolio} />
    //             </Paper>
    //         </Grid>
    //     </Grid>
    // );
    if (!portfolio || !portfolio.stocks) {
        return <Typography>Aucun portefeuille disponible</Typography>;
    }

    const totalValue = portfolio.stocks.reduce((sum, stock) => sum + stock.weight * stock.entryPrice, 0);

    return (
        <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
                <Paper elevation={3} style={{ padding: '20px' }}>
                    <Typography variant="h6">Portfolio Value</Typography>
                    <Typography variant="h4">${totalValue.toFixed(2)}</Typography>
                </Paper>
            </Grid>
            <Grid item xs={12} md={6}>
                <Paper elevation={3} style={{ padding: '20px' }}>
                    <Typography variant="h6">Portfolio Allocation</Typography>
                    {portfolio && portfolio.stocks && portfolio.stocks.length > 0 ? (
                        <PortfolioPieChart portfolio={portfolio} />
                    ) : (
                        <Typography>Aucune donnée disponible</Typography>
                    )}
                </Paper>
            </Grid>
            {/* Ajoutez d'autres widgets ici */}
        </Grid>
    );
}

export default Dashboard;